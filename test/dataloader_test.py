'''Test script for the dataloader.

To run: python -m test.dataloader_test
'''

import json
import unittest
from datasets import load_dataset
from transformers import AutoTokenizer

from src.dataloader import *

class DataLoaderTest(unittest.TestCase):
    data_filenames = {
        'train': 'test/data/test_dataset.csv',
        'validation': 'test/data/test_dataset.csv',
    } # user needs to make sure the input and output keys match that of the transforms
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    t5_tokenizer = AutoTokenizer.from_pretrained('t5-small')

    print('Loading dataset...')
    train_data = load_dataset('csv', data_files=data_filenames, split='train')
    causal_lm_dataset = CustomDataset(
        tokenizer,
        data = train_data,
        input_transforms = ['std'],
        conditioned_transforms = ['null'],
        max_length = 512,
        max_prompt_length = 256,
        truncation_mode = 'keep_start',
        use_causal_lm = True
    )

    conditional_generation_dataset = CustomDataset(
        t5_tokenizer,
        data = train_data,
        input_transforms = ['std'],
        conditioned_transforms = ['null'],
        max_length = 512,
        max_prompt_length = 256,
        use_causal_lm = False
    )

    def test_causal_lm_dataset(self):
        dataset = self.causal_lm_dataset
        self.assertEqual(3, len(dataset))

        input_data_after_transform = [
            {'input': "This is a simple test sentence.", 'label': 1},
            {'input': "This is another simple test sentence.", 'label': 2},
        ]
        cond_data_after_transform = [
            {'input': " ", 'label': 1},
            {'input': " ", 'label': 2},
        ]
        for i, ex in enumerate(input_data_after_transform):
            for k,v in ex.items():
                self.assertEqual(v, dataset.input_data[i][k])
        for i, ex in enumerate(cond_data_after_transform):
            for k,v in ex.items():
                self.assertEqual(v, dataset.conditioned_data[i][k])

    def test_causal_lm_dataset_tokenize(self):
        dataset = self.causal_lm_dataset
        tokens_input_0 = self.tokenizer(dataset.input_data[0]['input'])
        tokens_cond_0 = self.tokenizer(dataset.conditioned_data[0]['input'])
        tokens_label_0 = self.tokenizer(str(dataset.input_data[0]['label']))
        print(tokens_input_0)
        print(tokens_cond_0)
        print(tokens_label_0)
        # {'input_ids': [1, 910, 338, 263, 2560, 1243, 10541, 29889], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1]}
        # {'input_ids': [1, 259], 'attention_mask': [1, 1]}
        # {'input_ids': [1, 29871, 29896], 'attention_mask': [1, 1, 1]}

        print('============causal lm tokens============')
        print(dataset[0])
        # input/conditioned_input and label are concatenated to the input sequence
        # note that we do not use the bos token (1), and add the eos token (2) to the end of the input
        # {
        #     'input': 'This is a simple test sentence.',
        #     'conditioned': ' ',
        #     'output': '1',
        #     'input_prompt_input_ids': [910, 338, 263, 2560, 1243, 10541, 29889, 2],
        #     'input_prompt_attention_mask': [1, 1, 1, 1, 1, 1, 1, 1],
        #     'conditioned_prompt_input_ids': [259, 2],
        #     'conditioned_prompt_attention_mask': [1, 1],
        #     'input_seq_input_ids': [910, 338, 263, 2560, 1243, 10541, 29889, 2, 29871, 29896],
        #     'input_seq_attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        #     'input_seq_labels': [-100, -100, -100, -100, -100, -100, -100, -100, 29871, 29896],
        #     'conditioned_seq_input_ids': [259, 2, 29871, 29896],
        #     'conditioned_seq_attention_mask': [1, 1, 1, 1],
        #     'conditioned_seq_labels': [-100, -100, 29871, 29896]
        # }

    def test_cond_gen_dataset_tokenize(self):
        dataset = self.conditional_generation_dataset
        print('==========conditional generation tokens============')
        print(dataset[0])
        # {
        #     'input': 'This is a simple test sentence.',
        #     'conditioned': ' ',
        #     'output': '1',
        #     'input_input_ids': tensor([[    1,   910,   338,   263,  2560,  1243, 10541, 29889]]),
        #     'input_attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1]]),
        #     'conditioned_input_ids': tensor([[  1, 259]]),
        #     'conditioned_attention_mask': tensor([[1, 1]]),
        #     'output_labels': tensor([[    1, 29871, 29896]]),
        # }

    def test_dataloader(self):
        dataloader = get_dataloader(self.causal_lm_dataset)

        print('==========batched causal lm tokens============')
        for batch in dataloader:
            print(batch)
        # note that padding of the prompts are to the left and padding of the prompt+label sequences are to the right
        # {
        #     'input': ['This is a simple test sentence.', 'Here is yet another slightly more complex test example.', 'This is another simple test sentence.'],
        #     'conditioned': [' ', ' ', ' '],
        #     'output': ['1', '1', '2'],
        #     'input_prompt_input_ids': tensor([
        #         [    2,     2,     2,   910,   338,   263,  2560,  1243, 10541, 29889, 2],
        #         [ 2266,   338,  3447,  1790, 10029,   901,  4280,  1243,  1342, 29889, 2],
        #         [    2,     2,     2,   910,   338,  1790,  2560,  1243, 10541, 29889, 2]
        #     ]), 'input_prompt_attention_mask': tensor([
        #         [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        #         [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
        #     ]), 'conditioned_prompt_input_ids': tensor([
        #         [259,   2], [259,   2], [259,   2]
        #     ]), 'conditioned_prompt_attention_mask': tensor([
        #         [1, 1], [1, 1], [1, 1]
        #     ]), 'input_seq_input_ids': tensor([
        #         [  910,   338,   263,  2560,  1243, 10541, 29889,     2, 29871, 29896, 2,     2,     2],
        #         [ 2266,   338,  3447,  1790, 10029,   901,  4280,  1243,  1342, 29889, 2, 29871, 29896],
        #         [  910,   338,  1790,  2560,  1243, 10541, 29889,     2, 29871, 29906, 2,     2,     2]
        #     ]), 'input_seq_attention_mask': tensor([
        #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
        #     ]), 'input_seq_labels': tensor([
        #         [ -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100, 29871, 29896,  -100,  -100,  -100],
        #         [ -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100, 29871, 29896],
        #         [ -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100, 29871, 29906,  -100,  -100,  -100]
        #     ]), 'conditioned_seq_input_ids': tensor([
        #         [  259,     2, 29871, 29896],
        #         [  259,     2, 29871, 29896],
        #         [  259,     2, 29871, 29906]
        #     ]), 'conditioned_seq_attention_mask': tensor([
        #         [1, 1, 1, 1],
        #         [1, 1, 1, 1],
        #         [1, 1, 1, 1]
        #     ]), 'conditioned_seq_labels': tensor([
        #         [ -100,  -100, 29871, 29896],
        #         [ -100,  -100, 29871, 29896],
        #         [ -100,  -100, 29871, 29906]
        #     ])}

    def test_dataloader_cond_gen(self):
        dataloader = get_dataloader(self.conditional_generation_dataset)

        print('==========batched conditional generation tokens============')
        for batch in dataloader:
            print(batch)
        # note that eos tokens (1) are added to every sequence
        # {
        #     'input': ['This is another simple test sentence.', 'This is a simple test sentence.', 'Here is yet another slightly more complex test example.'],
        #     'conditioned': [' ', ' ', ' '],
        #     'output': ['2', '1', '1'],
        #     'input_input_ids': tensor([
        #         [100,   19,  430,  650,  794, 7142,    5,    1,    0,    0,    0]
        #         [100,   19,    3,    9,  650,  794, 7142,    5,    1,    0,    0],
        #         [947,   19,  780,  430, 3300,   72, 1561,  794,  677,    5,    1],
        #     ]), 'input_attention_mask': tensor([
        #         [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        #     ]), 'conditioned_input_ids': tensor([
        #         [1], [1], [1]
        #     ]), 'conditioned_attention_mask': tensor([
        #         [1], [1], [1]
        #     ]), 'output_labels': tensor([
        #         [204, 1],
        #         [209, 1],
        #         [209, 1]
        #     ])}


if __name__ == '__main__':
    unittest.main()

