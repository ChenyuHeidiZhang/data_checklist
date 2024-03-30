import re
import os
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

def get_last_checkpoint(folder):
    dirs = [f for f in os.scandir(folder) if f.is_dir()]
    if len(dirs)==0:
        return
    dirs.sort(key=os.path.getctime)
    name = dirs[-1].name
    return os.path.join(folder,name)

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    return preds, labels

def load_tokenizer(model_config, config):
    if model_config.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_config.tokenizer_name, use_fast=not config.use_slow_tokenizer,
            trust_remote_code=config.trust_remote_code, cache_dir=config.cache_dir
        )
    elif model_config.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_config.model_name_or_path, use_fast=not config.use_slow_tokenizer,
            trust_remote_code=config.trust_remote_code, cache_dir=config.cache_dir
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    if model_config.task_type == 'CAUSAL_LM': #and not model_config.tokenizer_name.startswith('mistralai'):
        tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.model_max_length = config.max_length

    return tokenizer

def load_transform_model(model_config, config):
    # Load the transform model, which needs to be a causal language model.
    # Currently only works on single GPU. Mixtral loading fails with Cuda OOM otherwise.
    config_ = AutoConfig.from_pretrained(model_config.model_name_or_path,
                                        trust_remote_code=config.trust_remote_code)

    if model_config.dtype == '4bits':  # used for Mixtral-8x7B
        model = AutoModelForCausalLM.from_pretrained(
            model_config.model_name_or_path,
            load_in_4bit=True,  # loads the model in GPU with 4-bit weights
            device_map="auto",
            config=config_,
            cache_dir=config.cache_dir,
        )
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model = AutoModelForCausalLM.from_pretrained(
            model_config.model_name_or_path,
            torch_dtype=getattr(torch, model_config.dtype),
            from_tf=bool(".ckpt" in model_config.model_name_or_path),
            config=config_,
            trust_remote_code=config.trust_remote_code,
            cache_dir=config.cache_dir,
        ).to(device)

    return model
