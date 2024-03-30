from src.utils import *

import torch
import json
import logging
import math
import os
import datasets
import transformers
import warnings
from omegaconf import OmegaConf, DictConfig

import numpy as np

from peft import LoraConfig, get_peft_model
from peft.utils.other import fsdp_auto_wrap_policy
from torch.nn import CrossEntropyLoss
from pathlib import Path
from accelerate import Accelerator
from accelerate.logging import get_logger
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    AutoConfig,
    get_scheduler,
    set_seed,
)
from huggingface_hub import Repository, create_repo

logger = get_logger(__name__)


class BasicTrainer(object):
    def __init__(self, config, exp_prefix, train_dataloader,
                 eval_dataloader, tokenizer, save_model_path):
        """Our basic trainer
        """
        self.config = config
        self.exp_prefix = exp_prefix  # input or conditioned
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.tokenizer = tokenizer
        self.run_dir = save_model_path
        self.no_decay = ["bias", "layer_norm.weight"]
        self.repo = None
        self.result = None
        self.model_cls = None  # to be initialized in each trainer class

    def load_model(self, conf):
        raise NotImplementedError

    def get_batch_loss(self, batch, model):
        raise NotImplementedError

    def eval_loop(self, model, accelerator, epoch, completed_steps):
        raise NotImplementedError

    def train(self):
        """Begin training, with periodic evaluation."""

        if not self.config.do_train:
            logging.info("Skipping training since `do_train` is set to False.")
            # assert get_last_checkpoint(
                # self.run_dir) is not None, f"Checkpoint must exist at {self.run_dir} if not doing training."
            return

        # Detecting last checkpoint.
        # we don't need to train the same model multiple times.
        last_checkpoint = None
        os.makedirs(self.run_dir, exist_ok=True)
        if not self.config.overwrite_output_dir:
            if len(os.listdir(self.run_dir)) > 0:
                warnings.warn(
                    f"Output directory ({self.run_dir}) already exists and is not empty. Training will be resumed or skipped if already complete. "
                    "Use --config.train.overwrite_output_dir to train from scratch.", RuntimeWarning)
            last_checkpoint = get_last_checkpoint(self.run_dir)

        # Set seed before initializing model.
        set_seed(self.config.seed)

        accelerator_log_kwargs = {}
        if self.config.wandb.enabled:
            accelerator = Accelerator(log_with="wandb",
                                      gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                                      **accelerator_log_kwargs)
            accelerator.init_trackers(project_name=self.config.wandb.project + "-" + self.run_dir.split("/")[-1],
                init_kwargs={"wandb": {'entity': self.config.wandb.entity,
                                       'config': OmegaConf.to_container(self.config)}})
        else:
            accelerator = Accelerator(gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                                      **accelerator_log_kwargs)

        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(accelerator.state, main_process_only=False)
        if accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_warning()
            # transformers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()

        # Handle the repository creation
        if accelerator.is_main_process:
            if self.config.push_to_hub:
                # Retrieve of infer repo_name
                repo_name = self.config.model.hub_model_id
                if repo_name is None:
                    repo_name = Path(self.run_dir).absolute().name
                # Create repo and retrieve repo_id
                repo_id = create_repo(repo_name, exist_ok=True, token=self.config.hub_token).repo_id
                # Clone repo locally
                self.repo = Repository(self.run_dir, clone_from=repo_id, token=self.config.hub_token)

                with open(os.path.join(self.run_dir, ".gitignore"), "w+") as gitignore:
                    if "step_*" not in gitignore:
                        gitignore.write("step_*\n")
                    if "epoch_*" not in gitignore:
                        gitignore.write("epoch_*\n")
        accelerator.wait_for_everyone()

        # Load pretrained model
        #
        # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
        # download model & vocab.

        config_ = AutoConfig.from_pretrained(self.config.model.model_name_or_path,
                                             trust_remote_code=self.config.trust_remote_code)

        model = self.load_model(config_)
        if self.config.model.use_lora:
            peft_config = LoraConfig(
                inference_mode=False, r=self.config.model.lora_config.r,
                lora_alpha=self.config.model.lora_config.lora_alpha,
                lora_dropout=self.config.model.lora_config.lora_dropout)
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

        # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
        # on a small vocab and want a smaller embedding size, remove this test.
        embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(self.tokenizer) > embedding_size:
            model.resize_token_embeddings(len(self.tokenizer))

        # TODO do we need to take care of this?
        # if model.config.decoder_start_token_id is None:
        #    raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

        # this setup is necessary to use FSDP with PEFT (https://huggingface.co/docs/peft/en/accelerate/fsdp)
        if getattr(accelerator.state, "fsdp_plugin", None) is not None:
            print("Using FSDP!!!!!!!!")
            accelerator.state.fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(model)

        # Prepare the model before initializing optimizer is recommended for FSDP.
        model = accelerator.prepare(model)
        # model, train_dataloader = accelerator.prepare(model, self.train_dataloader)  # DS requires passing in a dataloader

        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in self.no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in self.no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=float(self.config.learning_rate))
        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / self.config.gradient_accumulation_steps)
        if self.config.max_train_steps is None:
            self.config.max_train_steps = self.config.num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True

        lr_scheduler = get_scheduler(
            name=self.config.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.config.num_warmup_steps * self.config.gradient_accumulation_steps,
            num_training_steps=self.config.max_train_steps * self.config.gradient_accumulation_steps,
        )

        optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
            optimizer, self.train_dataloader, self.eval_dataloader, lr_scheduler
        )
        # eval_dataloader still not on device, let's do it separately
        self.eval_dataloader = accelerator.prepare(self.eval_dataloader)

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / self.config.gradient_accumulation_steps)
        if overrode_max_train_steps:
            self.config.max_train_steps = self.config.num_train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        self.config.num_train_epochs = math.ceil(self.config.max_train_steps / num_update_steps_per_epoch)

        # Figure out how many steps we should save the Accelerator states
        checkpointing_steps = self.config.checkpointing_steps
        if checkpointing_steps is not None and checkpointing_steps.isdigit():
            checkpointing_steps = int(checkpointing_steps)

        # Train!
        logger.info("***** Running training *****")

        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(self.config.max_train_steps), disable=not accelerator.is_local_main_process)
        completed_steps = 0
        starting_epoch = 0

        # Potentially load in the weights and states from a previous save
        if last_checkpoint:
            checkpoint_path = last_checkpoint
            path = os.path.basename(checkpoint_path)

            # Extract `epoch_{i}` or `step_{i}`
            training_difference = os.path.splitext(path)[0]

            if "epoch" in training_difference:
                starting_epoch = int(training_difference.replace("epoch_", "")) + 1
                resume_step = None
                completed_steps = starting_epoch * num_update_steps_per_epoch
            else:
                # need to multiply `gradient_accumulation_steps` to reflect real steps
                resume_step = int(training_difference.replace("step_", "")) * self.config.gradient_accumulation_steps
                starting_epoch = resume_step // len(train_dataloader)
                completed_steps = resume_step // self.config.gradient_accumulation_steps
                resume_step -= starting_epoch * len(train_dataloader)

            if starting_epoch >= self.config.num_train_epochs:
                warnings.warn(f"It looks like training was complete in a previous run. Skipping training!"
                              "Use --config.train.overwrite_output_dir to overcome.", RuntimeWarning)
                return

            accelerator.print(f"Trying to resume from checkpoint: {checkpoint_path}")
            accelerator.load_state(checkpoint_path)

        # update the progress_bar if load from checkpoint
        progress_bar.update(completed_steps)

        for epoch in range(starting_epoch, self.config.num_train_epochs):
            model.train()
            if last_checkpoint and epoch == starting_epoch and resume_step is not None:
                # We skip the first `n` batches in the dataloader when resuming from a checkpoint
                active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
            else:
                active_dataloader = train_dataloader
            for step, batch in enumerate(active_dataloader):
                with accelerator.accumulate(model):
                    loss = self.get_batch_loss(batch, model)
                    loss = loss / self.config.gradient_accumulation_steps
                    if step % 100 == 0:
                        logger.info(f"step {step}, loss: {loss:.3f}")
                        if self.config.wandb.enabled:
                            accelerator.log({"train_loss": loss}, step=completed_steps)

                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1

                if completed_steps >= self.config.max_train_steps:
                    break

            model.eval()
            stop_early = self.eval_loop(model, accelerator, epoch, completed_steps)
            if stop_early:
                logger.info("Early Stopping!")
                break

        if self.config.wandb.enabled:
            accelerator.end_training()

        if self.run_dir is not None:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                self.run_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                state_dict=accelerator.get_state_dict(model),  # TODO: do we need this line? (https://huggingface.co/docs/accelerate/en/usage_guides/deepspeed)
            )
            if accelerator.is_main_process:
                self.tokenizer.save_pretrained(self.run_dir)
                if self.config.push_to_hub:
                    self.repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)

                all_results = {f"eval_{k}": v for k, v in self.result.items()}
                with open(os.path.join(self.run_dir, "all_results.json"), "w") as f:
                    json.dump(all_results, f)

    def get_batch_logps_manual(
            self, logits: torch.FloatTensor, labels: torch.LongTensor, average_log_prob: bool = True,
            shift_logits: bool = True
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        assert logits.shape[:-1] == labels.shape
        labels = labels.clone()

        if shift_logits:
            labels = labels[:, 1:]  # .clone()
            logits = logits[:, :-1, :]
        loss_mask = (labels != -100)

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == -100] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)
