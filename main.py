'''Main entry point for the package (handles training and checklisting.)

Run with hydra configs:
python main.py +data=<data_config_name> +model=<model_config_name>
e.g. python main.py +data=cola +model=t5-small +transform_model=llama2-7b (optional)
e.g. accelerate launch --config_file accelerate_config.yaml main.py std_transform_func=std_shp null_transform_func=null_shp batch_size=4 +data=shp +model=t5-base
'''

from typing import List
import os
import logging

import hydra
from omegaconf import OmegaConf, DictConfig
from datasets import load_dataset, Dataset

from src.dataloader import CustomDataset, get_dataloader
from src.data_checklist import data_check_vinfo
from src.data_transforms import InputTransforms
from src.trainer_seq2seq import Seq2SeqTrainer
from src.trainer_clm import CLMTrainer
from src.format_shp import sample_shp_data_from_hf
from src import utils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

"""
Applicability Test:
Attribute att is feasibile if V_info(att(X) -> Y) > 0.
That is, it is possible to learn something using the given attribute.

Exclusivity Test:
Attribute att is exclusive if V_info(att'(X) -> Y) = 0.
That is, all V-usable information exists only in the given attribute.

Sufficiency Test:
Attribute att is sufficient if V_info(X -> Y | att(X)) = 0.
That is, all V-usable information that X contains about Y is also contained in the given attribute.

Necessity Test:
Attribute att is necessary if V_info(X -> Y | att'(X)) > 0.
That is, some V-usable information exists only in the given attribute. Otherwise, the attribute is redundant.

Regular V-info:
V_info(X -> Y) is the regular V-info.
"""

# Below is a reference for the transforms needed for each check type.
# Two models need to be trained for each check on the following inputs, respectively:
# (1) conditioned out variable Z: the input from the former transform, and
# (2) input X: the input that combines the former and the latter transformed results.
# Entropy of Y|(conditioned Z) - Y|(input X) givens the test v-info.
# Note that att or att_inv can consist of multiple chained transforms.
CHECK_TYPE_TO_DATA_TYPES = {
    'applicability': ['null', 'att'],
    'inapplicability': ['null', 'att'],
    'exclusivity': ['null', 'att_inv'],
    'non-exclusivity': ['null', 'att_inv'],
    'sufficiency': ['att', 'std'],
    'insufficiency': ['att', 'std'],
    'necessity': ['att_inv', 'std'],
    'redundancy': ['att_inv', 'std'],
    'regular_vinfo': ['null', 'std'],
}


def load_data_by_split(data_config, split_name, cache_dir=None):
    '''Returns the loaded dataset and the name of the dataset for naming the trained model.'''
    if data_config.hf_data_name:
        logger.info(f'Loading {split_name} data from {data_config.hf_data_name}.')
        if 'hf_subset_name' in data_config:
            input_data = load_dataset(data_config.hf_data_name, split=split_name, name=data_config.hf_subset_name, cache_dir=cache_dir)
        elif 'hf_datadir_name' in data_config:
            input_data = load_dataset(data_config.hf_data_name, split=split_name, data_dir=data_config.hf_datadir_name, cache_dir=cache_dir)
        else:
            input_data = load_dataset(data_config.hf_data_name, split=split_name, cache_dir=cache_dir)
        data_name = data_config.hf_data_name.split('/')[-1]
    else:
        logger.info(f'Loading data from {data_config.data_filenames[split_name]}.')
        data_files = OmegaConf.to_container(data_config.data_filenames, resolve=True)
        input_data = load_dataset(data_config.data_type, data_files=data_files, split=split_name)
        data_name = data_config.data_filenames[split_name].split('/')[-1].split('.')[0]
    # cut input data to 100 examples
    # input_data = Dataset.from_dict(input_data[:100])
    return input_data, data_name


def get_dataloader_with_configs(config, tokenizer, input_data, input_transforms, conditioned_transforms, transforms_class, shuffle=True):
    dataset = CustomDataset(
        tokenizer,
        data = input_data,
        input_transforms = input_transforms,
        conditioned_transforms = conditioned_transforms,
        transforms_class = transforms_class,
        max_length = config.max_length,
        max_prompt_length = config.max_prompt_length,
        use_causal_lm = config.model.task_type == 'CAUSAL_LM'
    )

    return get_dataloader(
        dataset,
        per_device_batch_size=config.batch_size,
        shuffle=shuffle
    )


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):

    # Resolve hydra references, e.g. so we don't re-compute the run directory
    OmegaConf.resolve(config)

    missing_keys: Set[str] = OmegaConf.missing_keys(config)
    if missing_keys:
        raise ValueError(f"Got missing keys in config:\n{missing_keys}")

    logger.info(OmegaConf.to_yaml(config))

    # check task type
    assert config.model.task_type in ['SEQ_2_SEQ_LM', 'CAUSAL_LM'], 'Only CausalLM or Seq2SeqLM are supported for now.'

    # initialize tokenizer
    tokenizer = utils.load_tokenizer(config.model, config)

    # load data
    assert config.data.hf_data_name or config.data.data_filenames, "Must provide either hf_data_name or data_filenames"
    assert not (config.data.hf_data_name and config.data.data_filenames), "Must provide either hf_data_name or data_filenames, not both"
    if config.data.data_filenames:
        assert config.data.data_type, "Must provide data_type if using data_filenames"

    if config.data.hf_data_name == 'stanfordnlp/shp':
        data_name = 'shp'
        train_data, eval_data = sample_shp_data_from_hf(config.data.ratio_thresh, config.data.examples_per_post, data_dir=config.data.hf_datadir_name)
    else:
        eval_data, data_name = load_data_by_split(config.data, config.data.eval_split_name, cache_dir=config.cache_dir)
        if config.do_train or config.eval_on_train:
            train_data, data_name = load_data_by_split(config.data, config.data.train_split_name, cache_dir=config.cache_dir)
        else:
            # dummy train data to save time
            train_data = Dataset.from_dict(eval_data[:10])

    if 'hf_datadir_name' in config.data and config.data.hf_datadir_name:
        data_name += '_' + config.data.hf_datadir_name

    # save configs
    config_path = os.path.join(config.model_output_dir, data_name, 'config.yaml')
    if not os.path.exists(os.path.dirname(config_path)):
        os.makedirs(os.path.dirname(config_path))
    with open(config_path, 'w') as f:
        OmegaConf.save(config, f)

    # initialize transforms
    logger.info('Initializing transforms...')
    transform_tokenizer = tokenizer
    transform_model, transform_model_config = None, None
    if 'transform_model' in config:
        logger.info(f'Loading transform model from {config.transform_model}.')
        transform_model_config=config.transform_model
        transform_tokenizer = utils.load_tokenizer(transform_model_config, config)
        transform_model = utils.load_transform_model(transform_model_config, config)
    else:
        logger.info('No transform model specified.')

    transforms_class = InputTransforms(
        config.data.input_key,
        config.data.output_key,
        config.data.additional_input_keys,
        transform_tokenizer,
        transform_model,
        transform_model_config
    )

    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
        logger.info('Output directory does not exist, created it.')

    # TODO: give the file better names to avoid overwriting
    results_log_file = os.path.join(config.output_dir, f'{data_name}_results_log.txt')

    # train models and run tests
    for check_type in config.check_types:
        assert check_type in CHECK_TYPE_TO_DATA_TYPES, f"Check type {check_type} not supported"
        logger.info(f'Running {check_type} test.')

        conditioned_transforms, input_transforms = [config.null_transform_func], [config.std_transform_func]
        if check_type == 'applicability' or check_type == 'inapplicability':
            input_transforms = config.attribute_func
        elif check_type == 'exclusivity' or check_type == 'non-exclusivity':
            input_transforms = config.inverse_attribute_func
        elif check_type == 'sufficiency' or check_type == 'insufficiency':
            conditioned_transforms = config.attribute_func
        elif check_type == 'necessity' or check_type == 'redundancy':
            conditioned_transforms = config.inverse_attribute_func

        cond_transform_id = '-'.join(conditioned_transforms)
        inp_transform_id = '-'.join(input_transforms)

        donot_concat = any([t in transforms_class.no_concat_cond_transforms for t in conditioned_transforms])
        if 'null' not in cond_transform_id.split('_') and not donot_concat:
            inp_transform_id = f'{inp_transform_id}_{cond_transform_id}'

        cond_model_save_path = os.path.join(config.model_output_dir, data_name, cond_transform_id)
        inp_model_save_path = os.path.join(config.model_output_dir, data_name, inp_transform_id)
        model_save_paths = {'input': inp_model_save_path, 'conditioned': cond_model_save_path}

        shuffle=False if config.eval_on_train else True
        train_dataloader = get_dataloader_with_configs(
            config, tokenizer, train_data, input_transforms, conditioned_transforms, transforms_class, shuffle=shuffle)
        # do not shuffle the eval dataloader to get consistent output across input and cond models
        eval_dataloader = get_dataloader_with_configs(
            config, tokenizer, eval_data, input_transforms, conditioned_transforms, transforms_class, shuffle=False)

        # We formulate the task as sequence generation; currently only causal lm or seq2seq is supported.
        trainers = {}
        for p in ['input', 'conditioned']:
            save_path = model_save_paths[p]
            logger.info(f'Training model for the {p} case and will save to {save_path}!')
            if config.model.task_type == 'CAUSAL_LM':
                trainer = CLMTrainer(config,
                                     p,
                                     train_dataloader,
                                     eval_dataloader,
                                     tokenizer,
                                     save_path)
            else:
                trainer = Seq2SeqTrainer(config,
                                         p,
                                         train_dataloader,
                                         eval_dataloader,
                                         tokenizer,
                                         save_path)

            trainer.train()
            trainers[p] = trainer

        # compute v-infos
        suffix_datasplit = 'train' if config.eval_on_train else ''
        out_fn = os.path.join(config.output_dir, f'{data_name}_{check_type}_pvi_{suffix_datasplit}.csv')
        logger.info(f'Computing v-infos for {check_type}...')

        dataloader = train_dataloader if config.eval_on_train else eval_dataloader
        data_check_vinfo(
            dataloader,
            trainers['input'],
            trainers['conditioned'],
            out_fn,
            check_type,
            results_log_file,
            config
        )


if __name__ == '__main__':
    main()
