from src.trainer import BasicTrainer
from src.utils import *

import torch
import json
import logging
import math
import os
import transformers
import evaluate

from torch.nn import CrossEntropyLoss
from pathlib import Path
from accelerate import Accelerator
from accelerate.logging import get_logger
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM

logger = get_logger(__name__)

class CLMTrainer(BasicTrainer):
    def __init__(self, config, exp_prefix, train_dataloader,
                 eval_dataloader, tokenizer, save_model_path):
        """A trainer subclass for causal language modeling
        """
        super().__init__(config, exp_prefix, train_dataloader,
                         eval_dataloader, tokenizer, save_model_path)
        self.model_cls = AutoModelForCausalLM

    def load_model(self, config_):
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model.model_name_or_path,
            torch_dtype=getattr(torch, self.config.model.dtype),
            from_tf=bool(".ckpt" in self.config.model.model_name_or_path),
            config=config_,
            low_cpu_mem_usage=self.config.model.low_cpu_mem_usage,
            trust_remote_code=self.config.trust_remote_code,
            cache_dir=self.config.cache_dir,
        )

        return model

    def get_batch_loss(self, batch, model, reduction='mean'):
        lm_logits = model(batch[f'{self.exp_prefix}_seq_input_ids'],
                          attention_mask=batch[f'{self.exp_prefix}_seq_attention_mask']).logits.to(torch.float32)
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = batch[f'{self.exp_prefix}_seq_labels'][..., 1:].contiguous()
        loss_fct = CrossEntropyLoss(reduction=reduction)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return loss

    def eval_loop(self, model, accelerator, epoch, completed_steps):
        '''Computes perplexity on the eval set. Returns whether training should stop early as the loss no longer decreases.'''

        losses = []
        for step, batch in enumerate(self.eval_dataloader):
            with torch.no_grad():
                loss = self.get_batch_loss(batch, model)
                losses.append(accelerator.gather_for_metrics(loss.repeat(self.config.batch_size)))
        losses = torch.cat(losses)
        eval_loss = torch.mean(losses)
        try:
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")

        logger.info(f"epoch {epoch}: perplexity: {perplexity:.3f} , eval_loss: {eval_loss:.3f}")
        if self.config.wandb.enabled:
            accelerator.log({"perplexity": perplexity, "eval_loss": eval_loss}, step=completed_steps)

        if self.result and self.result['loss'] <= eval_loss:
            # early stopping
            return True

        self.result = {'perplexity': round(float(perplexity), 3), 'loss': round(float(eval_loss), 3)}

        if self.config.push_to_hub and epoch < self.config.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                self.run_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                self.tokenizer.save_pretrained(self.run_dir)
                self.repo.push_to_hub(
                    commit_message=f"Training in progress epoch {epoch}", blocking=False, auto_lfs_prune=True
                )

        if self.config.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if self.run_dir is not None:
                output_dir = os.path.join(self.run_dir, output_dir)
            accelerator.save_state(output_dir)

        return False

    def run_inference(self, batch, model, accelerator, class_ids=[], class_labels=[]):
        outputs = accelerator.unwrap_model(model).generate(
            batch[f'{self.exp_prefix}_prompt_input_ids'],
            attention_mask=batch[f'{self.exp_prefix}_prompt_attention_mask'],
            max_new_tokens=self.config.max_output_length
        )
        predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions = [pred[len(batch[f'input'][i]):] for i, pred in enumerate(predictions)]

        entropies = []

        return predictions, entropies

    def compute_entropy(self, batch, model):
        logits = model(
            batch[f'{self.exp_prefix}_seq_input_ids'],
            attention_mask=batch[f'{self.exp_prefix}_seq_attention_mask'],
        ).logits.to(torch.float32)

        labels = batch[f'{self.exp_prefix}_seq_labels']
        logps = self.get_batch_logps_manual(logits, labels, average_log_prob=True)
        entropies = -logps / np.log(2)  # convert to bits

        return entropies.cpu().numpy()

