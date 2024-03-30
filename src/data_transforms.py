'''Specify input transforms here.'''
import re
import logging
import string
import torch
from torch.nn.utils.rnn import pad_sequence
import pysbd
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob

import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('brown')  # Download the Brown Corpus
from nltk.corpus import brown, stopwords
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
import string
import pycld2 as cld2
import regex as re

import numpy as np
np.random.seed(0)

import os
os.environ['PYTHONHASHSEED'] = str(0)

from src import utils
from src.format_shp import PreferencePrompt, RESPONSE_TOKEN_1, RESPONSE_TOKEN_2

nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('spacytextblob')

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


IN_FIELD = 'input'
OUT_FIELD = 'label'

class InputTransforms:
    '''A class that groups input transforms.
    '''

    def __init__(
        self,
        input_key='sentence1',
        output_key='label',
        additional_input_keys=None,
        tokenizer=None,
        transform_model=None,
        transform_model_config=None,
    ):
        self.transform_name_to_func = {
            'std': self.standard_transform,
            'null': self.null_transform,
            # Specify your own standard transform if necessary
            'std_nli': self.nli_standard_transform,
            'std_shp': self.shp_standard_transform,
            'null_shp': self.shp_null_transform,
            'std_shp_align': self.shp_standard_alignment_transform,
            'null_shp_align': self.shp_null_alignment_transform,
            'std_hh_align': self.anthropic_standard_alignment_transform,
            'null_hh_align': self.anthropic_null_alignment_transform,
            'std_hh_pref': self.anthropic_standard_preference_transform,
            'null_hh_pref': self.anthropic_null_preference_transform,
            'std_uf_pref': self.ultrafeedback_standard_preference_transform,
            'null_uf_pref': self.ultrafeedback_null_preference_transform,
            'null_uf_align': self.ultrafeedback_null_alignment_transform,
            # Add attribute functions here
            'snli_overlap': self.nli_overlap_transform,
            'snli_overlap_tokens': self.nli_overlap_tokens_transform,
            'dwmw_vocab': self.dwmw_vocab_transform,
            'inv_dwmw_vocab': self.dwmw_vocab_inverse_transform,
            'dwmw_sentiment': self.dwmw_sentiment_transform,
            'grammar': self.grammar_transform,  # model-based transform
            'shp_humor': self.shp_batched_humor_transform,  # model-based transform
            'shp_word_length': self.shp_word_length_transform,
            'inv_shp_word_length': self.shp_word_length_inverse_transform,
            # 'shp_overlap': self.shp_overlap_transform,
            # 'inv_shp_overlap': self.shp_overlap_inverse_transform,
            'shp_word_complexity': self.shp_word_complexity_transform,
            'hh_vocab': self.anthropic_vocab_transform,
            'inv_hh_vocab': self.anthropic_vocab_inverse_transform,
            'hh_pref_vocab': self.anthropic_preference_vocab_transform,
            'inv_hh_pref_vocab': self.anthropic_preference_vocab_inverse_transform,
            'hh_pref_length': self.anthropic_preference_word_length_transform,
            'inv_hh_pref_length': self.anthropic_preference_word_length_inverse_transform,
            'uf_non_eng': self.ultrafeedback_non_english_transform,
            'inv_uf_non_eng': self.ultrafeedback_non_english_inverse_transform,
            'uf_noneng_align': self.ultrafeedback_non_eng_alignment_transform,
            'uf_score': self.ultrafeedback_score_delta_transform,
        }

        # Conditioned transforms whose result do not need to be concatenated with the input, because the input already contains that information.
        self.no_concat_cond_transforms = ['inv_shp_word_length']

        # Transforms that require batched model-based transform
        self.batch_transforms = ['grammar', 'shp_humor']

        self.input_key = input_key
        self.output_key = output_key
        self.additional_input_keys = additional_input_keys  # list of additional input keys
        self.tokenizer = tokenizer
        if not self.tokenizer.mask_token:
            self.tokenizer.mask_token = "<mask>"

        if transform_model:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.transform_batch_size = transform_model_config.batch_size
            self.transform_model = transform_model  # should have been put to the correct device

        # used for shp
        self.segmenter = pysbd.Segmenter(language="en", clean=False)

        # potentially offensive words were manually selected for DWMW
        self.bad_words = [ 'nigga', 'niggas', 'niggah', 'niggahs', 'hoe', 'hoes', 'bitch', 'bitches', 'whitey', 'white trash', 'cracker', 'crackers', 'beaner', 'beaners',
            'pussy', 'pussies', 'fag', 'fags', 'faggot', 'faggots', 'ho', 'hos', 'redneck', 'rednecks', 'porn', 'fuck', 'fucks', 'fucker', 'fuckers', 'motherfucker', 
            'motherfuckers', 'nigger', 'niggers', 'coon', 'coons', 'niggaz', 'nig', 'nigs', 'slut', 'sluts', 'wigger', 'wiggers', 'fucked', 'fucking', 'wigga', 'wiggas',
            'retard', 'retards', 'retarded' ] 

        self.bad_words_long = []
        with open('src/profane_words.txt', 'r') as f:  # 1383 bad words from https://www.cs.cmu.edu/~biglou/resources/
            for line in f.readlines():
                self.bad_words_long.append(line.strip())

        self.stop_words = set(stopwords.words('english'))
        # Create a frequency distribution of words in the Brown Corpus
        self.brown_freq_dist = FreqDist(brown.words())


    # TODO: clear this up
    # multiple transforms? multiple keys?
    def transform_wrapper(self, example, input_transform_names):
        '''Takes in a data instance from a dataset, and transforms it using the input transforms.
        '''
        input_key_store = self.input_key
        for transform in input_transform_names:
            example = self.transform_name_to_func[transform](example)
            self.input_key = IN_FIELD  # when using multiple transforms, future transforms use IN_FIELD as key
        self.input_key = input_key_store
        assert IN_FIELD in example, 'Transform function must save result to the input field.'
        if not OUT_FIELD in example: 
            # only apply output key if OUT_FIELD is not already assigned during transform
            example[OUT_FIELD] = example[self.output_key]
        return example

    def standard_transform(self, example):
        example[IN_FIELD] = example[self.input_key]
        return example

    def null_transform(self, example):
        example[IN_FIELD] = " "
        return example

    # Please specify your own transform functions here.

    def nli_overlap_transform(self, example):
        hypothesis_tokens = self.tokenizer.tokenize(example['hypothesis'])
        overlap_tokens = [ t for t in hypothesis_tokens if t in self.tokenizer.tokenize(example['premise']) ]
        overlap = len(overlap_tokens) / len(hypothesis_tokens)

        if overlap >= 0.75:
            msg = "HIGH OVERLAP"
        elif overlap >= 0.5:
            msg = "MEDIUM OVERLAP"
        elif overlap >= 0.25:
            msg = "LOW OVERLAP"
        else:
            msg = "NO OVERLAP"

        example[IN_FIELD] = f"{msg}."	
        return example

    def nli_overlap_tokens_transform(self, example):
        hypothesis_tokens = self.tokenizer.tokenize(example['hypothesis'])
        premise_tokens = self.tokenizer.tokenize(example['premise']) 
        overlap = set(hypothesis_tokens) & set(premise_tokens)
        hypothesis = " ".join([ (t if t in overlap else self.tokenizer.mask_token) for t in hypothesis_tokens ])
        premise = " ".join([ (t if t in overlap else self.tokenizer.mask_token) for t in premise_tokens ])
        example[IN_FIELD] = f"PREMISE: {premise} HYPOTHESIS: {hypothesis}"	
        return example


    def _vocab_transform(self, in_text, bad_words):
        # extracts offensive words
        pattern = re.compile(rf"(?!\b({'|'.join(bad_words)})\b)\b[^ ]+\b", re.IGNORECASE)
        in_text = re.sub(pattern, "", in_text)
        in_text = in_text.translate(str.maketrans('', '', string.punctuation))
        in_text = in_text.strip()

        if in_text == "":
            in_text = ' ' #using only empty string can yield problems

        return in_text

    def _vocab_inverse_transform(self, in_text, bad_words):
        # extracts non-offensive words
        pattern = re.compile(rf"\b({'|'.join(bad_words)})\b(?!\b[^ ]+\b)", re.IGNORECASE)
        in_text = re.sub(pattern, "", in_text)
        in_text = in_text.translate(str.maketrans('', '', string.punctuation))
        in_text = in_text.strip()

        if in_text == "":
            in_text = ' ' #using only empty string can yield problems

        return in_text

    def dwmw_vocab_transform(self, example):
        example[IN_FIELD] = self._vocab_transform(example[self.input_key], self.bad_words)
        return example

    def dwmw_vocab_inverse_transform(self, example):
        example[IN_FIELD] = self._vocab_inverse_transform(example[self.input_key], self.bad_words)
        return example

    def dwmw_sentiment_transform(self, example):
        polarity = nlp(example[self.input_key])._.polarity 

        if -0.10 <= polarity <= 0.10:
            sentiment = 'neutral'
        elif polarity > 0.10:
            sentiment = 'positive'
        else:
            sentiment = 'negative'

        example[IN_FIELD] = sentiment

        return example

    def nli_standard_transform(self, example):
        example[IN_FIELD] = f"PREMISE: {example['premise']} HYPOTHESIS: {example['hypothesis']}"
        return example

    def batched_model_based_transform(self, examples, prompt='', max_length=128):
        orig_input_texts = examples[self.input_key]
        input_texts = [prompt.format(ex_in) for ex_in in orig_input_texts]
        # logger.info(input_texts)
        input_tokens = self.tokenizer(input_texts, padding=True, return_tensors='pt')
        outputs = self.transform_model.generate(
            input_tokens['input_ids'].to(self.device),
            attention_mask=input_tokens['attention_mask'].to(self.device),
            max_length=max_length,
        )
        preds = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        examples[IN_FIELD] = [pred[len(input_texts[i]):].strip() for i, pred in enumerate(preds)]
        # logger.info(examples[IN_FIELD])
        return examples

    def grammar_transform(self, examples):
        prompt = 'Does the following sentence violate any grammatical rule? If so, specify which rule it violates.\n' + \
        'Sentence: "I jumps on the bed." This sentence violates the subject-verb agreement rule.\n' + \
        'Sentence: "Sara enjoyed her dinner." This sentence does not violate any grammatical rule.\n' + \
        'Sentence: "The boy loves we." This sentence has the wrong object form for "we".\n' + \
        'Sentence: "{}" '
        ''
        return self.batched_model_based_transform(examples, prompt=prompt, max_length=256)


    def _shp_format_preference_prompt(
        self, example,
        fields=('history', 'human_ref_A', 'human_ref_B'),
        check_slack=True, truncate_responses=True, return_prompt=False
    ):
        prompt = PreferencePrompt("", example[fields[1]], example[fields[2]])

        if not check_slack:
            prompt.post = PreferencePrompt.clean_text(example[fields[0]])
            return str(prompt) if not return_prompt else prompt

        slack = self.tokenizer.model_max_length - len(self.tokenizer(str(prompt)).input_ids)

        if slack > 0:
            sentences = []
            for s in self.segmenter.segment(PreferencePrompt.clean_text(example[fields[0]])):
                slack -= len(self.tokenizer(s).input_ids)

                if slack > 0:
                    sentences.append(s)

            prompt.post = "".join(sentences)
        elif truncate_responses:
            # if responses are longer than max length, truncate them
            # add segments of both resposnes in parallel until we reach the max length
            total_num_tokens = 0
            seg_A = self.segmenter.segment(PreferencePrompt.clean_text(example[fields[1]]))
            seg_B = self.segmenter.segment(PreferencePrompt.clean_text(example[fields[2]]))
            sentences_A = []
            sentences_B = []
            for i in range(len(seg_A)):
                total_num_tokens += len(self.tokenizer(seg_A[i]).input_ids)
                sentences_A.append(seg_A[i])
                if i < len(seg_B):
                    total_num_tokens += len(self.tokenizer(seg_B[i]).input_ids)
                    sentences_B.append(seg_B[i])
                if total_num_tokens > self.tokenizer.model_max_length:
                    break
            # continue with seg B if there are more tokens left
            if total_num_tokens <= self.tokenizer.model_max_length and len(seg_B) > len(sentences_B):
                for i in range(len(sentences_B), len(seg_B)):
                    total_num_tokens += len(self.tokenizer(seg_B[i]).input_ids)
                    sentences_B.append(seg_B[i])
                    if total_num_tokens > self.tokenizer.model_max_length:
                        break

            prompt.response_a = "".join(sentences_A)
            prompt.response_b = "".join(sentences_B)

        return str(prompt) if not return_prompt else prompt

    def shp_standard_transform(self, example, check_slack=True):
        example[IN_FIELD] = self._shp_format_preference_prompt(example, check_slack=check_slack)
        example[OUT_FIELD] = RESPONSE_TOKEN_1 if example["labels"] == 1 else RESPONSE_TOKEN_2
        return example

    def shp_null_transform(self, example):
        example[IN_FIELD] = " "
        example[OUT_FIELD] = RESPONSE_TOKEN_1 if example["labels"] == 1 else RESPONSE_TOKEN_2
        return example

    def shp_batched_humor_transform(self, examples):
        # Prompt formatted for Mixtral-8x7B-Instruct-v0.1

        USER_MESSAGE_1 = 'QUESTION: Tell me a joke. RESPONSE: Why did the scarecrow win an award? Because he was outstanding in his field. HUMOR SCORE:'
        BOT_MESSAGE_1 = '4'
        USER_MESSAGE_2 = 'QUESTION: Say something. RESPONSE: I find the best way to start a conversation is with a joke. HUMOR SCORE:'
        BOT_MESSAGE_2 = '2'
        PROMPT = 'Rate how humorous the given response is with a single number, on a scale of 1 to 5.\n'

        def tokenize(text):
            return self.tokenizer.encode(text, add_special_tokens=False)

        BOS_ID = self.tokenizer.bos_token_id
        EOS_ID = self.tokenizer.eos_token_id
        instr = [BOS_ID] + \
            tokenize("[INST]") + tokenize(USER_MESSAGE_1) + tokenize("[/INST]") + tokenize(BOT_MESSAGE_1) + [EOS_ID] + \
            tokenize("[INST]") + tokenize(USER_MESSAGE_2) + tokenize("[/INST]") + tokenize(BOT_MESSAGE_2) + [EOS_ID] + \
            tokenize("[INST]") + tokenize(PROMPT)

        ex_histories = []
        for his, res_a, res_b in zip(examples['history'], examples['human_ref_A'], examples['human_ref_B']):
            formatted_prompt = self._shp_format_preference_prompt(
                {'history': his, 'human_ref_A': res_a, 'human_ref_B': res_b}, return_prompt=True)
            ex_histories.append(formatted_prompt.post)

        # ex_histories = examples['history']

        def get_humor_score(res_field='human_ref_A'):
            ex_responses = examples[res_field]

            messages = [f'QUESTION: {his} RESPONSE: {res} HUMOR SCORE:' for his, res in zip(ex_histories, ex_responses)]
            instructions = [
                torch.tensor(instr + tokenize(msg) + tokenize("[/INST]")) for msg in messages]
            attention_mask = [torch.ones_like(ins) for ins in instructions]

            instructions = pad_sequence(
                instructions, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            attention_mask = pad_sequence(
                attention_mask, batch_first=True, padding_value=0)

            outputs = self.transform_model.generate(
                instructions.to(self.device),
                attention_mask=attention_mask.to(self.device),
                max_new_tokens=20,
            )
            preds = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            preds_nums = [0] * len(ex_histories)
            for i, pred in enumerate(preds):
                pred_str = pred.split("[/INST]")[-1]
                pred_num = re.findall('[1-5]', pred_str)
                if len(pred_num) == 0:
                    logger.warning(f"Invalid humor score: {pred_str}")
                    continue
                preds_nums[i] = int(pred_num[0])
            return preds_nums

        humor_scores_A = get_humor_score(res_field='human_ref_A')
        humor_scores_B = get_humor_score(res_field='human_ref_B')

        examples[IN_FIELD] = [f'Humor scores: A: {score_A}, B: {score_B}' for score_A, score_B in zip(humor_scores_A, humor_scores_B)]
        logger.info(examples[IN_FIELD])

        examples[OUT_FIELD] = [RESPONSE_TOKEN_1 if label == 1 else RESPONSE_TOKEN_2 for label in examples["labels"]]

        return examples

    def shp_word_length_transform(self, example):
        example[IN_FIELD] = len(example['human_ref_A'].split()) - len(example['human_ref_B'].split())
        example[OUT_FIELD] = RESPONSE_TOKEN_1 if example["labels"] == 1 else RESPONSE_TOKEN_2
        return example

    def shp_word_length_inverse_transform(self, example, check_slack=True):
        # Repeat the shorter sentence as many times as possible to while still being shorter than the longer sentence.
        # NOTE: For the necessity transform, it doesn't make sense to concatenate this with the standard input.
        A_tokens = example['human_ref_A'].split()
        B_tokens = example['human_ref_B'].split()
        if len(A_tokens) == 0: A_tokens = [' ']
        if len(B_tokens) == 0: B_tokens = [' ']
        if len(A_tokens) < len(B_tokens):
            example['human_ref_A'] = " ".join(A_tokens * (len(B_tokens) // len(A_tokens)))
        elif len(B_tokens) < len(A_tokens):
            example['human_ref_B'] = " ".join(B_tokens * (len(A_tokens) // len(B_tokens)))

        return self.shp_standard_transform(example, check_slack=check_slack)

    def shp_overlap_transform(self, example):
        # Mask out words that are not in both sentences. Note that this transform still contains length information.
        A_tokens = word_tokenize(example['human_ref_A'])
        B_tokens = word_tokenize(example['human_ref_B'])
        overlap = set(A_tokens) & (set(B_tokens))
        human_ref_A = " ".join([(t if t in overlap else self.tokenizer.mask_token) for t in A_tokens])
        human_ref_B = " ".join([(t if t in overlap else self.tokenizer.mask_token) for t in B_tokens])
        example['human_ref_A'] = human_ref_A
        example['human_ref_B'] = human_ref_B

        return self.shp_standard_transform(example)

    def shp_overlap_inverse_transform(self, example):
        # Mask out words that are in both sentences. Note that this transform still contains length information.
        A_tokens = word_tokenize(example['human_ref_A'])
        B_tokens = word_tokenize(example['human_ref_B'])
        overlap = set(A_tokens) & (set(B_tokens))
        human_ref_A = " ".join([(t if t not in overlap else self.tokenizer.mask_token) for t in A_tokens])
        human_ref_B = " ".join([(t if t not in overlap else self.tokenizer.mask_token) for t in B_tokens])
        example['human_ref_A'] = human_ref_A
        example['human_ref_B'] = human_ref_B

        return self.shp_standard_transform(example)

    def _get_word_complexity(self, sent, raw_score=True):
        complexity_scores = []
        tokenized_sent = word_tokenize(sent)
        for word in tokenized_sent:
            if not word.lower() in self.stop_words and word not in string.punctuation:
                f = self.brown_freq_dist.freq(word.lower())
                if f > 0:  # word exists in our dict
                    complexity_scores.append(-np.log(f))
                else:
                    complexity_scores.append(10)  # regard unseen ones as complex word
        score = np.mean(complexity_scores)
        if raw_score:
            return round(score, 3)
            # return score

        if score <= 9:
            return 'low complexity'
        elif 9 < score <= 9.5:
            return 'medium low complexity'
        elif 9.5 < score <= 10:
            return 'medium high complexity'
        else:
            return 'high complexity'

    def shp_word_complexity_transform(self, example):
        complexity_A = self._get_word_complexity(example['human_ref_A'])
        complexity_B = self._get_word_complexity(example['human_ref_B'])
        example[IN_FIELD] = f"Word complexity {RESPONSE_TOKEN_1}: {complexity_A}. Word complexity {RESPONSE_TOKEN_2}: {complexity_B}."
        # example[IN_FIELD] = f"Word complexity diff: {round((complexity_A - complexity_B) * 100, 1)}"
        example[OUT_FIELD] = RESPONSE_TOKEN_1 if example["labels"] == 1 else RESPONSE_TOKEN_2

        return example

    def shp_standard_alignment_transform(self, example):
        # this can be combined with transforms like DWMW vocab transform to get the bad words from the input.
        example[IN_FIELD] = example['history']
        example[OUT_FIELD] = example['human_ref_A'] if example["labels"] == 1 else example['human_ref_B']

        return example

    def shp_null_alignment_transform(self, example):
        # this can be combined with transforms like DWMW vocab transform to get the bad words from the input.
        example[IN_FIELD] = " "
        example[OUT_FIELD] = example['human_ref_A'] if example["labels"] == 1 else example['human_ref_B']

        return example

    def anthropic_standard_alignment_transform(self, example):
        query, response = example['chosen'].rsplit('Assistant: ', 1)
        example[IN_FIELD] = query.strip() + 'Assistant: '
        example[OUT_FIELD] = response

        return example

    def anthropic_null_alignment_transform(self, example):
        _, response = example['chosen'].rsplit('Assistant: ', 1)
        example[IN_FIELD] = ' '
        example[OUT_FIELD] = response

        return example

    def anthropic_vocab_transform(self, example):
        query, response = example['chosen'].rsplit('Assistant: ', 1)
        remaining_words = self._vocab_transform(query.strip(), self.bad_words_long)
        example[IN_FIELD] = remaining_words + 'Assistant: '
        example[OUT_FIELD] = response
        return example

    def anthropic_vocab_inverse_transform(self, example):
        query, response = example['chosen'].rsplit('Assistant: ', 1)
        remaining_words = self._vocab_inverse_transform(query.strip(), self.bad_words_long)
        example[IN_FIELD] = remaining_words + 'Assistant: '
        example[OUT_FIELD] = response
        return example

    def _format_anthropic_as_shp_example(self, example):
        # so that we can use the same transform functions for both SHP and anthropic
        query_c, response_c = example['chosen'].rsplit('Assistant: ', 1)
        query_r, response_r = example['rejected'].rsplit('Assistant: ', 1)
        # note: around 1% of the data has longer/shorter (query) preambles for chosen and rejected

        if hash(query_c) % 2 == 0:
            example['human_ref_A'] = response_c
            example['human_ref_B'] = response_r
            example['labels'] = 1
        else:
            example['human_ref_A'] = response_r
            example['human_ref_B'] = response_c
            example['labels'] = 0

        example['history'] = query_c.strip()
        return example

    def anthropic_standard_preference_transform(self, example):
        query_c, response_c = example['chosen'].rsplit('Assistant: ', 1)
        query_r, response_r = example['rejected'].rsplit('Assistant: ', 1)
        # note: around 1% of the data has longer/shorter (query) preambles for chosen and rejected

        if hash(query_c) % 2 == 0:
            prompt = PreferencePrompt(query_c.strip(), response_c, response_r)
            response_token = RESPONSE_TOKEN_1  # the first response (response A) is better
        else:
            prompt = PreferencePrompt(query_c.strip(), response_r, response_c)
            response_token = RESPONSE_TOKEN_2

        example[IN_FIELD] = str(prompt)
        example[OUT_FIELD] = response_token
        return example

    def anthropic_null_preference_transform(self, example):
        query_c, _ = example['chosen'].rsplit('Assistant: ', 1)
        example[IN_FIELD] = " "
        # null would always be used as conditioned transform, where the output field doesn't matter (since we use the output field from the input transform)
        example[OUT_FIELD] = RESPONSE_TOKEN_1 if hash(query_c) % 2 == 0 else RESPONSE_TOKEN_2
        return example

    def anthropic_preference_vocab_transform(self, example):
        # Applies bad word transform on the responses to choose from.
        query_c, response_c = example['chosen'].rsplit('Assistant: ', 1)
        remaining_response_c = self._vocab_transform(response_c, self.bad_words_long)

        query_r, response_r = example['rejected'].rsplit('Assistant: ', 1)
        remaining_response_r = self._vocab_transform(response_r, self.bad_words_long)

        if hash(query_c) % 2 == 0:
            prompt = PreferencePrompt(query_c.strip(), remaining_response_c, remaining_response_r)
            response_token = RESPONSE_TOKEN_1  # the first response (response A) is better
        else:
            prompt = PreferencePrompt(query_c.strip(), remaining_response_r, remaining_response_c)
            response_token = RESPONSE_TOKEN_2

        example[IN_FIELD] = str(prompt)
        example[OUT_FIELD] = response_token
        return example

    def anthropic_preference_vocab_inverse_transform(self, example):
        # Applies bad word transform on the responses to choose from.
        query_c, response_c = example['chosen'].rsplit('Assistant: ', 1)
        remaining_response_c = self._vocab_inverse_transform(response_c, self.bad_words_long)

        query_r, response_r = example['rejected'].rsplit('Assistant: ', 1)
        remaining_response_r = self._vocab_inverse_transform(response_r, self.bad_words_long)

        if hash(query_c) % 2 == 0:
            prompt = PreferencePrompt(query_c.strip(), remaining_response_c, remaining_response_r)
            response_token = RESPONSE_TOKEN_1
        else:
            prompt = PreferencePrompt(query_c.strip(), remaining_response_r, remaining_response_c)
            response_token = RESPONSE_TOKEN_2
        
        example[IN_FIELD] = str(prompt)
        example[OUT_FIELD] = response_token
        return example

    def anthropic_preference_word_length_transform(self, example):
        example_shp = self._format_anthropic_as_shp_example(example)
        return self.shp_word_length_transform(example_shp)

    def anthropic_preference_word_length_inverse_transform(self, example):
        example_shp = self._format_anthropic_as_shp_example(example)
        return self.shp_word_length_inverse_transform(example_shp, check_slack=False)


    def ultrafeedback_standard_preference_transform(self, example, get_words_func=None):
        if get_words_func:
            query = get_words_func(example['prompt'])
            response_c = get_words_func(example['chosen'][-1]['content'])
            response_r = get_words_func(example['rejected'][-1]['content'])
        else:
            query = example['prompt']
            response_c = example['chosen'][-1]['content']
            response_r = example['rejected'][-1]['content']

        if hash(example['prompt']) % 2 == 0:
            # prompt = PreferencePrompt(query, response_c, response_r)
            prompt = self._shp_format_preference_prompt(
                {'history': query, 'human_ref_A': response_c, 'human_ref_B': response_r})
            response_token = RESPONSE_TOKEN_1
        else:
            # prompt = PreferencePrompt(query, response_r, response_c)
            prompt = self._shp_format_preference_prompt(
                {'history': query, 'human_ref_A': response_r, 'human_ref_B': response_c})
            response_token = RESPONSE_TOKEN_2

        example[IN_FIELD] = str(prompt)
        example[OUT_FIELD] = response_token
        return example

    def ultrafeedback_null_preference_transform(self, example):
        query = example['prompt']
        example[IN_FIELD] = " "
        example[OUT_FIELD] = RESPONSE_TOKEN_1 if hash(query) % 2 == 0 else RESPONSE_TOKEN_2
        return example

    def _get_eng_or_non_eng(self, text, get_eng=False):
        text = text.replace('\n', ' ')
        # according to https://github.com/aboSamoor/polyglot/issues/71
        RE_BAD_CHARS = re.compile(r"[\p{Cc}\p{Cs}]+")
        def remove_bad_chars(text):
            return RE_BAD_CHARS.sub("", text)
        text = remove_bad_chars(text)  # Cc category

        text = text.encode('utf-8')
        _, _, _, lang_vecs = cld2.detect(text, returnVectors=True)
        new_text = b''
        for vec in lang_vecs:
            if (vec[-1] == 'en') == get_eng:
                new_text += text[vec[0]:vec[0]+vec[1]]
        new_text = new_text.decode('utf-8')
        return new_text

    def ultrafeedback_non_english_transform(self, example):
        # mask out all english words
        get_words_func = lambda text: self._get_eng_or_non_eng(text, get_eng=False)

        return self.ultrafeedback_standard_preference_transform(
            example, get_words_func=get_words_func)

    def ultrafeedback_non_english_inverse_transform(self, example):
        # mask out all non-english words
        get_words_func = lambda text: self._get_eng_or_non_eng(text, get_eng=True)

        return self.ultrafeedback_standard_preference_transform(
            example, get_words_func=get_words_func)

    def ultrafeedback_null_alignment_transform(self, example):
        example[IN_FIELD] = " "
        example[OUT_FIELD] = example['chosen'][-1]['content']
        return example

    def ultrafeedback_non_eng_alignment_transform(self, example):
        get_words_func = lambda text: self._get_eng_or_non_eng(text, get_eng=False)

        query = get_words_func(example['prompt'])

        example[IN_FIELD] = query
        example[OUT_FIELD] = example['chosen'][-1]['content']

        return example

    def ultrafeedback_score_delta_transform(self, example):
        # to test whether text contains more information than the score delta
        if hash(example['prompt']) % 2 == 0:
            # A: chosen, B: rejected
            score_delta = example['score_chosen'] - example['score_rejected']
            response_token = RESPONSE_TOKEN_1
        else:
            # A: rejected, B: chosen
            score_delta = example['score_rejected'] - example['score_chosen']
            response_token = RESPONSE_TOKEN_2

        example[IN_FIELD] = str(score_delta)
        example[OUT_FIELD] = response_token
        return example
