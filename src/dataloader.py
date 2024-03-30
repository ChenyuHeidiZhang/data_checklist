'''
Given a huggingface dataset or a data file, and a data transform (null, some attribute, or standard),
construct a dataloader. Each example would contain tokenized inputs and labels.
'''

from typing import List, Dict, Union, Callable
import functools
import copy
import logging
import pandas as pd

from accelerate import Accelerator
from datasets import load_dataset
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from transformers import default_data_collator, DataCollatorWithPadding

from src.data_transforms import InputTransforms, IN_FIELD, OUT_FIELD

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class CustomDataset(Dataset):
    def __init__(self,
        tokenizer,
        data,
        input_transforms = ['std'],
        conditioned_transforms = ['null'],
        transforms_class = None,
        max_length = 512,
        max_prompt_length = 256,
        truncation_mode = 'keep_start',  # truncation for the prompt; only supported for causal LM
        use_causal_lm = True,
    ):
        '''
        tokenizer: The tokenizer to use.
        data: dataset loaded with load_dataset from huggingface or a data file.
        input_transforms: The names of transformations to apply to X. Default is the standard transformation which keeps the original data.
        conditioned_transforms: The names of transformations to apply to the conditioned variable, which defaults to empty.
        use_causal_lm: default True, which uses causal language modeling. If False, uses conditional generation.
        '''

        # self.input_data = data
        # self.conditioned_data = copy.deepcopy(self.input_data)

        self.tokenizer = tokenizer
        self.input_transforms = input_transforms
        self.conditioned_transforms = conditioned_transforms
        self.transform_cls = transforms_class
        if not self.transform_cls:
            self.transform_cls = InputTransforms(tokenizer=self.tokenizer)

        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.truncation_mode = truncation_mode
        self.use_causal_lm = use_causal_lm

        self.input_data = self.apply_transform(data, input_transforms)
        self.conditioned_data = self.apply_transform(data, conditioned_transforms)

    def _load_transformed_data_csv_if_exists(self, data_path):
        try:
            df = pd.read_csv(data_path)
            df = pd.DataFrame(df)
            return Dataset.from_pandas(df)
            # return load_dataset('csv', data_files=data_path)
        except FileNotFoundError:
            return None

    def apply_transform(self, data, transform_names):
        # after transform, each data instance would have two keys: 'input' and 'label'
        transform_funcs = functools.partial(
            self.transform_cls.transform_wrapper, input_transform_names=transform_names)

        logger.info('Applying data transforms...')

        if set(self.transform_cls.batch_transforms).intersection(transform_names):
            if transform_names == ['shp_humor']:
                transformed_data = self._load_transformed_data_csv_if_exists('shp_data_humor_transformed.csv')
                if transformed_data and len(transformed_data) == len(data):
                    return transformed_data

            bz = self.transform_cls.transform_batch_size
            transformed_data = data.map(transform_funcs, batched=True, batch_size=bz)

            # Distributed inference not working yet. TODO: Check out https://huggingface.co/docs/accelerate/main/en/usage_guides/distributed_inference
            # accelerator = Accelerator()

            # dataloader = DataLoader(data, batch_size=self.transform_cls.transform_batch_size)
            # self.transform_cls.transform_model, dataloader = accelerator.prepare(
            #     self.transform_cls.transform_model, dataloader)
            # self.transform_cls.transform_model.eval()

            # transformed_data = {}
            # for batch in dataloader:
            #     with torch.no_grad():
            #         transformed_batch = transform_funcs(batch)
            #     for k,v in transformed_batch.items():
            #         if k not in transformed_data:
            #             transformed_data[k] = []
            #         transformed_data[k].extend(v)

            if transform_names == ['shp_humor']:
                # save the 'input' and 'label' keys of the transformed data
                data_df = transformed_data.to_pandas()[['history', 'human_ref_A', 'human_ref_B', IN_FIELD, OUT_FIELD]]
                with open('shp_data_humor_transformed.csv', 'w') as f:
                    data_df.to_csv(f, index=False)
        else:
            transformed_data = data.map(transform_funcs)

        return transformed_data

    def __len__(self):
        return len(self.input_data)

    def format_causal_lm_tokens(self, inputs, conditioned_inputs, outputs):
        '''Adapted from the DPO repo. Tokenize a single batch element.
    
        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation.

        We also create the labels for the input and conditioned input, which are of length equal to
            the sum of the length of the input and the response, with -100 for the input tokens.
        '''
        inputs_tokens = self.tokenizer(inputs, add_special_tokens=False)
        conditioned_inputs_tokens = self.tokenizer(conditioned_inputs, add_special_tokens=False)
        outputs_tokens = self.tokenizer(outputs, add_special_tokens=False)

        # assert self.tokenizer.eos_token_id not in inputs_tokens['input_ids']
        # assert self.tokenizer.eos_token_id not in conditioned_inputs_tokens['input_ids']
        # assert self.tokenizer.eos_token_id not in outputs_tokens['input_ids']

        inputs_tokens['input_ids'].append(self.tokenizer.eos_token_id)
        inputs_tokens['attention_mask'].append(1)
        conditioned_inputs_tokens['input_ids'].append(self.tokenizer.eos_token_id)
        conditioned_inputs_tokens['attention_mask'].append(1)

        # if combined sequence is too long, truncate the prompt
        if len(conditioned_inputs_tokens['input_ids']) + len(outputs_tokens) > self.max_length:
            if self.truncation_mode == 'keep_start':
                conditioned_inputs_tokens = {k: v[:self.max_prompt_length] for k, v in conditioned_inputs_tokens.items()}
                inputs_tokens = {k: v[:self.max_prompt_length] for k, v in inputs_tokens.items()}
            elif self.truncation_mode == 'keep_end':
                conditioned_inputs_tokens = {k: v[-self.max_prompt_length:] for k, v in conditioned_inputs_tokens.items()}
                inputs_tokens = {k: v[-self.max_prompt_length:] for k, v in inputs_tokens.items()}
            else:
                raise ValueError(f'Unknown truncation mode: {self.truncation_mode}')

        # if that's still too long, truncate the response
        if len(conditioned_inputs_tokens['input_ids']) + len(outputs_tokens) > self.max_length:
            outputs_tokens = {k: v[:self.max_length - self.max_prompt_length] for k, v in outputs_tokens.items()}

        # Create labels
        inputs_sequence_tokens = {k: inputs_tokens[k] + outputs_tokens[k] for k in inputs_tokens}
        conditioned_sequence_tokens = {k: conditioned_inputs_tokens[k] + outputs_tokens[k] for k in conditioned_inputs_tokens}
        inputs_sequence_tokens['labels'] = inputs_sequence_tokens['input_ids'][:]
        inputs_sequence_tokens['labels'][:len(inputs_tokens['input_ids'])] = [-100] * len(inputs_tokens['input_ids'])
        conditioned_sequence_tokens['labels'] = conditioned_sequence_tokens['input_ids'][:]
        conditioned_sequence_tokens['labels'][:len(conditioned_inputs_tokens['input_ids'])] = [-100] * len(conditioned_inputs_tokens['input_ids'])

        return inputs_tokens, conditioned_inputs_tokens, inputs_sequence_tokens, conditioned_sequence_tokens


    def format_conditional_generation_tokens(self, inputs, conditioned_inputs, outputs):
        input_tokens = self.tokenizer(
            inputs, truncation=True, max_length=self.max_prompt_length, add_special_tokens=True, return_tensors=None
        )  # returns 1D list for each key
        conditioned_input_tokens = self.tokenizer(
            conditioned_inputs, truncation=True, max_length=self.max_prompt_length, add_special_tokens=True, return_tensors=None
        )
        output_tokens = self.tokenizer(
            outputs,
            truncation=True,
            max_length=self.max_length-len(conditioned_input_tokens['input_ids']),
            add_special_tokens=True,
            return_tensors=None
        )
        return input_tokens, conditioned_input_tokens, output_tokens


    def __getitem__(self, idx):
        """Adapted from DPO repo. Tokenize a single batch element.
    
        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
            in case the prompt + chosen or prompt + rejected responses is/are too long. First
            we truncate the prompt; if we're still too long, we truncate the chosen/rejected.
        
        We also create the labels for the chosen/rejected responses, which are of length equal to
            the sum of the length of the prompt and the chosen/rejected response, with -100 for the
            prompt tokens.
        """

        # convert to str to format everything as seq2seq
        inputs = str(self.input_data[idx][IN_FIELD])
        conditioned_inputs = str(self.conditioned_data[idx][IN_FIELD])
        donot_concat = any([t in self.transform_cls.no_concat_cond_transforms for t in self.conditioned_transforms])
        if conditioned_inputs.strip() != "" and not donot_concat:
            # inputs += " " + conditioned_inputs.strip()
            inputs = conditioned_inputs.strip() + " " + inputs
        outputs = str(self.input_data[idx][OUT_FIELD])

        batch = {}
        batch['input'] = inputs
        batch['conditioned'] = conditioned_inputs
        batch['output'] = outputs

        if self.use_causal_lm:
            inputs_tokens, conditioned_tokens, inputs_sequence_tokens, conditioned_sequence_tokens = self.format_causal_lm_tokens(
                inputs, conditioned_inputs, outputs)
            for k, toks in {
                'input_prompt': inputs_tokens,
                'conditioned_prompt': conditioned_tokens,
                'input_seq': inputs_sequence_tokens,
                'conditioned_seq': conditioned_sequence_tokens,
            }.items():
                for type_key, tokens in toks.items():
                    if type_key == 'token_type_ids':
                        continue
                    batch[f'{k}_{type_key}'] = tokens
        else:
            inputs_tokens, conditioned_tokens, outputs_tokens = self.format_conditional_generation_tokens(
                inputs, conditioned_inputs, outputs)
            for k, toks in {'input': inputs_tokens, 'conditioned': conditioned_tokens}.items():
                for type_key, tokens in toks.items():
                    if type_key == 'token_type_ids':
                        continue
                    batch[f'{k}_{type_key}'] = tokens
            batch['output_labels'] = outputs_tokens.input_ids

        return batch


def get_collate_fn(tokenizer) -> Callable[[List[Dict]], Dict[str, Union[List, torch.Tensor]]]:
    """Adapted from the DPO repo. Returns a collate function for the given tokenizer.

       The collate function takes a list of examples (dicts, where values are lists of
         ints [tokens] or strings [the original texts]) and returns a batch of examples,
         PyTorch tensors padded to the maximum length. Strings are passed through."""
    def collate_fn(batch):
        # first, pad everything to the same length
        padded_batch = {}
        for k in batch[0].keys():
            if k.endswith('_input_ids') or k.endswith('_attention_mask') or k.endswith('_labels'):
                if 'prompt' in k:  # adapted from https://stackoverflow.com/questions/73256206
                    to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                else:
                    to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                if k.endswith('_input_ids'):
                    padding_value = tokenizer.pad_token_id
                elif k.endswith('_labels'):
                    padding_value = -100
                elif k.endswith('_attention_mask'):
                    padding_value = 0
                else:
                    raise ValueError(f"Unexpected key in batch '{k}'")

                padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                if 'prompt' in k:  # for the prompt, flip back so padding is on left side
                    padded_batch[k] = padded_batch[k].flip(dims=[1])
            else:
                padded_batch[k] = [ex[k] for ex in batch]

        return padded_batch
    return collate_fn


def get_dataloader(
    dataset,
    per_device_batch_size=8,
    shuffle=True,
):
    data_collator = get_collate_fn(dataset.tokenizer)

    dataloader = DataLoader(
        dataset, shuffle=shuffle, collate_fn=data_collator, batch_size=per_device_batch_size
    )

    return dataloader
