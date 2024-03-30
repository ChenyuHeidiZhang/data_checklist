from src.trainer import BasicTrainer
from src.utils import *

import torch
import json
import logging
import math
import os
import transformers
import evaluate
import numpy as np

from torch.nn import CrossEntropyLoss
from pathlib import Path
from accelerate import Accelerator
from accelerate.logging import get_logger
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM

logger = get_logger(__name__)

class Seq2SeqTrainer(BasicTrainer):
    def __init__(self, config, exp_prefix, train_dataloader,
                 eval_dataloader, tokenizer, save_model_path):
        """A trainer subclass for seq2seq tasks
        """
        super().__init__(config, exp_prefix, train_dataloader,
                         eval_dataloader, tokenizer, save_model_path)

        self.model_cls = AutoModelForSeq2SeqLM
        self.metric = evaluate.load(self.config.model.metric)
        self.no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]

    def load_model(self, config_):
        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model.model_name_or_path,
            from_tf=bool(".ckpt" in self.config.model.model_name_or_path),
            config=config_,
            trust_remote_code=self.config.trust_remote_code,
            cache_dir=self.config.cache_dir,
        )
            # torch_dtype=getattr(torch, self.config.model.dtype),

        return model

    def get_batch_loss(self, batch, model, reduction='mean'):
        labels = batch['output_labels']
        lm_logits = model(batch[f'{self.exp_prefix}_input_ids'],
                          attention_mask=batch[f'{self.exp_prefix}_attention_mask'],
                          labels=labels).logits.to(torch.float32)

        loss_fct = CrossEntropyLoss(ignore_index=-100, reduction=reduction)
        loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        return loss

    def eval_loop(self, model, accelerator, epoch, completed_steps):
        gen_kwargs = {
            "max_length": self.config.max_output_length,
            "num_beams": self.config.model.num_beams,
        }
        for step, batch in enumerate(self.eval_dataloader):
            with torch.no_grad():
                generated_tokens = accelerator.unwrap_model(model).generate(
                    batch[f'{self.exp_prefix}_input_ids'],
                    attention_mask=batch[f'{self.exp_prefix}_attention_mask'],
                    **gen_kwargs,
                )

                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=self.tokenizer.pad_token_id
                )
                labels = batch['output_labels']

                # TODO check with dataloader?
                if not self.config.pad_to_max_length:
                    # If we did not pad to max length, we need to pad the labels too
                    labels = accelerator.pad_across_processes(labels, dim=1, pad_index=self.tokenizer.pad_token_id)

                generated_tokens, labels = accelerator.gather_for_metrics((generated_tokens, labels))
                generated_tokens = generated_tokens.cpu().numpy()
                labels = labels.cpu().numpy()

                # Replace -100 in the labels as we can't decode them.
                labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]
                decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                # TODO: use the original label sequence in the dataloader
                decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

                decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
                self.metric.add_batch(
                    predictions=decoded_preds,
                    references=decoded_labels,
                )
        new_result = self.metric.compute(use_stemmer=True)
        new_result = {k: round(v * 100, 4) for k, v in new_result.items()}
        logger.info(new_result)
        if self.config.wandb.enabled:
            accelerator.log(new_result, step=completed_steps)

        if self.result:
            if all([v >= new_result[k] for k, v in self.result.items()]):
                # early stopping
                return True

        self.result = new_result

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
            accelerator.save_state(output_dir, safe_serialization=False)

        return False

    def run_inference(self, batch, model, accelerator, class_ids=None, class_labels=None):
        # example class_ids: [[71], [272]]; class_labels: ['A', 'B']
        input_ids = batch[f'{self.exp_prefix}_input_ids']
        attention_mask = batch[f'{self.exp_prefix}_attention_mask']
        outputs = accelerator.unwrap_model(model).generate(
            input_ids,
            attention_mask=attention_mask,
            return_dict_in_generate=True, output_scores=True,
            max_new_tokens=self.config.max_output_length
        )
        predictions = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)

        entropies = []
        # if cls_label_ids is not empty, gather scores from the label_ids and compute pvi from there
        if class_ids and class_labels:
            for i in range(len(batch['output'])):
                total_score = 0
                label_score = 0
                for cls_label, cls_id in zip(class_labels, class_ids):
                    score = np.mean([outputs.scores[0][i, t].item() for t in cls_id])
                    score = np.exp(score)
                    total_score += score
                    if batch['output'][i] == cls_label:
                        label_score = score
                entropies.append(-np.log2(label_score) + np.log2(total_score))  # - log p(y)

        return predictions, entropies


    def compute_entropy(self, batch, model):
        labels = batch['output_labels']
        # replace special token id's of the labels by -100 so it's ignored by the loss
        labels[labels == self.tokenizer.eos_token_id] = -100
        input_ids = batch[f'{self.exp_prefix}_input_ids']
        attention_mask = batch[f'{self.exp_prefix}_attention_mask']
        logits = model(
            input_ids,
            attention_mask=attention_mask,
            labels = labels
        ).logits.to(torch.float32)
        logps = self.get_batch_logps_manual(logits, labels, average_log_prob=True, shift_logits=False)

        entropies = -logps / np.log(2)  # convert to bits

        return entropies.cpu().numpy()
