import logging
import os
from typing import List
from tqdm import tqdm
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoConfig
from accelerate import Accelerator, DistributedType
from peft import PeftModel
from peft.utils.other import fsdp_auto_wrap_policy

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_lora_model(trainer, config):
    base_model_name_or_path = config.model.model_name_or_path
    model_config = AutoConfig.from_pretrained(base_model_name_or_path)

    model = trainer.model_cls.from_pretrained(base_model_name_or_path, config=model_config, cache_dir=config.cache_dir)
    model = PeftModel.from_pretrained(model, trainer.run_dir)
    return model

def data_check_vinfo(
    eval_dataloader,
    input_trainer,
    cond_trainer,
    out_fn,
    check_type,
    results_log_file,
    config
):
    entropies = {'input': [], 'conditioned': []}
    predicted_labels = {'input': [], 'conditioned': []}
    input_raw = {'input': [], 'conditioned': []}
    references_labels = []

    do_inference = config.do_inference

    class_labels, class_ids = None, None
    if config.gather_class_logits:
        class_labels = list(config.class_labels)
        class_ids = input_trainer.tokenizer(class_labels, add_special_tokens=False)['input_ids']

    for trainer in [input_trainer, cond_trainer]:
        if config.model.use_lora:
            model = load_lora_model(trainer, config)
        else:
            # config_ = AutoConfig.from_pretrained(trainer.run_dir)
            # model = trainer.load_model(config_, model_name_or_path=trainer.run_dir)
            model = trainer.model_cls.from_pretrained(trainer.run_dir)

        accelerator = Accelerator()
        if getattr(accelerator.state, "fsdp_plugin", None) is not None:
            # assert not config.model.use_lora, 'Peft does not work with FSDP for inference at the moment. Use DeepSpeed instead.'
            # do_inference = False

            accelerator.state.fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(model)

        model, eval_dataloader_prepped = accelerator.prepare(model, eval_dataloader)

        model.eval()

        for batch in tqdm(eval_dataloader_prepped):
            with torch.no_grad():
                if config.gather_class_logits or do_inference:
                    preds, H = trainer.run_inference(batch, model, accelerator, class_ids, class_labels)
                    # print(preds, H)
                if not config.gather_class_logits:
                    H = trainer.compute_entropy(batch, model)

            references = batch['output']

            prefix = trainer.exp_prefix
            entropies[prefix].extend(accelerator.gather_for_metrics(H))
            if prefix == 'input':  # only add references once
                references_labels.extend(accelerator.gather_for_metrics(references))
            if do_inference:
                predicted_labels[prefix].extend(accelerator.gather_for_metrics(preds))
            input_raw[prefix].extend(accelerator.gather_for_metrics(batch[prefix]))


    if accelerator.is_main_process:
        data_dict = {
            'x_entropies': entropies['input'],
            'cond_entropies': entropies['conditioned'],
            'x_raw': input_raw['input'],
            'cond_raw': input_raw['conditioned'],
            'references_labels': references_labels,
        }
        if do_inference:
            data_dict['x_predicted_labels'] = predicted_labels['input']
            data_dict['cond_predicted_labels'] = predicted_labels['conditioned']

        data_df = pd.DataFrame(data_dict)

        data_df['PVI'] = data_df['cond_entropies'] - data_df['x_entropies']

        if out_fn:
            data_df.to_csv(out_fn)

        mean_pvi = data_df['PVI'].mean()
        logger.info(f"{check_type} test, mean PVI: {mean_pvi}")
        if check_type in ['feasibility', 'non-exclusivity', 'insufficiency', 'necessity', 'regular_vinfo']:
            criterion = mean_pvi > config.test_margin
        elif check_type in ['infeasibility', 'exclusivity', 'sufficiency', 'redundancy']:
            criterion = abs(mean_pvi) < config.test_margin

        if criterion:
            logger.info('Test passes.')
        else:
            logger.info('Test fails.')

        with open(results_log_file, 'a') as f:
            f.write(f"{check_type} test, mean PVI: {mean_pvi}\n")
            if criterion:
                f.write('Test passes.\n')
            else:
                f.write('Test fails.\n')
