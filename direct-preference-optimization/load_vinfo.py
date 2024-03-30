import os
import re
import json
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import pysbd
from transformers import AutoTokenizer

RESPONSE_TOKEN_1 = 'A'
RESPONSE_TOKEN_2 = 'B'
POST_TOKEN = "Context"
QUERY_TOKEN = "Question"

segmenter = pysbd.Segmenter(language="en", clean=False)
t5_tokenizer = AutoTokenizer.from_pretrained("t5-base")
t5_tokenizer.model_max_length = 512

class PreferencePrompt(object):
    """
    A class for formatting prompts.
    """
    def __init__(self, post, response_a, response_b):
        self.post = PreferencePrompt.clean_text(post)
        self.response_a = PreferencePrompt.clean_text(response_a)
        self.response_b = PreferencePrompt.clean_text(response_b)

    def __str__(self):
        prompt = (
            f"{POST_TOKEN}: " + self.post + 
            f" Response {RESPONSE_TOKEN_1}: " + self.response_a + 
            f" Response {RESPONSE_TOKEN_2}: " + self.response_b +
            f" {QUERY_TOKEN}: Which response is better? Response"
        )
        return prompt

    @staticmethod
    def clean_text(text: str) -> str:
        return text.replace("\n", " ")

    @staticmethod
    def from_text(text: str) -> object:
        match = re.split(f'{POST_TOKEN}:|Response {RESPONSE_TOKEN_1}:|Response {RESPONSE_TOKEN_2}:|{QUERY_TOKEN}:', text)
        
        if len(match) < 5:
            raise Exception(f"{text} not matched")
        else:
            return PreferencePrompt(match[1].strip(), match[2].strip(), match[3].strip())

def shp_format_preference_prompt(example, tokenizer=t5_tokenizer, check_slack=True):
    prompt = PreferencePrompt("", example["human_ref_A"], example["human_ref_B"])

    if not check_slack:
        prompt.post = PreferencePrompt.clean_text(example["history"])
        return str(prompt)

    slack = tokenizer.model_max_length - len(tokenizer(str(prompt)).input_ids)

    if slack > 0:
        sentences = []
        for s in segmenter.segment(PreferencePrompt.clean_text(example["history"])):
            slack -= len(tokenizer(s).input_ids)

            if slack > 0:
                sentences.append(s)

        prompt.post = "".join(sentences)

    return str(prompt)


def anthropic_format_alignment_prompt(example):
    query, response = example['chosen'].rsplit('Assistant: ', 1)
    return query.strip() + 'Assistant: '


def anthropic_format_preference_prompt(example):
    query_c, response_c = example['chosen'].rsplit('Assistant: ', 1)
    query_r, response_r = example['rejected'].rsplit('Assistant: ', 1)
    # note: around 1% of the data has preambles of different lengths for chosen and rejected

    # if hash(query_c) % 2 == 0:
    prompt = PreferencePrompt(query_c.strip(), response_c, response_r)
    # else:
    prompt2 = PreferencePrompt(query_c.strip(), response_r, response_c)

    # return str(prompt)
    # prompt_set1 = (prompt.post, prompt.response_a, prompt.response_b)
    # prompt_set2 = (prompt.post, prompt.response_b, prompt.response_a)
    # return prompt_set1, prompt_set2
    return str(prompt), str(prompt2)


def load_shp_wl_sufficiency_pvi():
    filename = '../checklist_out/shp_length_sufficiency_pvi.csv'
    # load data and build a dict that maps from the prompt to the PVI
    df = pd.read_csv(filename)
    sufficiency_pvi = {}
    for i, row in df.iterrows():
        prompt = row['x_raw'][:-len(row['cond_raw'])].strip()
        sufficiency_pvi[prompt] = float(row['PVI'])

    return sufficiency_pvi

def load_hh_harmless_vinfo():
    # filename = '../checklist_out/hh-rlhf_regular_vinfo_pvi-llama2.csv'  # alignment pvi
    filename = '../checklist_out3/hh-harmless_regular_vinfo_pvi_train.csv'  # preference pvi

    # load data and build a dict that maps from the prompt to the PVI
    df = pd.read_csv(filename)
    harmless_pvi = {}
    for i, row in df.iterrows():
        # prompt = PreferencePrompt.from_text(row['x_raw'])
        # prompt_set = (prompt.post, prompt.response_a, prompt.response_b)
        prompt_set = row['x_raw'].strip()
        harmless_pvi[prompt_set] = float(row['PVI'])

    # filters < 0 pvis prompts: 42536 -> 35464 (83%)

    return harmless_pvi


def load_uf_noneng_feasibility():
    filename = '../checklist_out3/ultrafeedback_binarized_noneng_feasibility_pvi_train.csv'
    # load data and build a dict that maps from the prompt to the PVI
    df = pd.read_csv(filename)
    noneng_feasibility_pvi = {}
    TEMPLATE = "Context:  Response A:  Response B:  Question: Which response is better? Response"
    for i, row in df.iterrows():  # the index is for the unshuffed dataset
        if row['x_raw'] == TEMPLATE:
            continue  # skip cases where there is no non-eng words
        noneng_feasibility_pvi[i] = float(row['PVI'])

    # mean_pvi = np.mean(list(noneng_feasibility_pvi.values()))
    # mean_pvi = 0  # remove everything with noneng words where pvi is below 0
    mean_pvi = 10  # remove everything with noneng words

    print('mean pvi:', mean_pvi)  # 0.00503
    print('num noneng:', len(noneng_feasibility_pvi))  # 17718
    print('num noneng < mean:', sum([1 for pvi in noneng_feasibility_pvi.values() if pvi < mean_pvi]))  # 9289; total uf: 61128 prompts (61135 prefs)
    return noneng_feasibility_pvi, mean_pvi

def load_uf_score_sufficiency(perc=0.5):
    # filename = '../checklist_out3/ultrafeedback_binarized_insufficiency_pvi_train_new.csv'
    filename = '../checklist_out4/ultrafeedback_binarized_insufficiency_pvi_train_flant5.csv'

    df = pd.read_csv(filename)
    score_sufficiency_pvi = {}
    for i, row in df.iterrows():  # the index is for the unshuffed dataset
        score_sufficiency_pvi[i] = float(row['PVI'])
        # set pvi to -1 if the score diff is 0, so that they are removed
        if row['cond_raw'] == 0:
            score_sufficiency_pvi[i] = -1

    # mislabeled examples: text + score has much less information than score alone (very negative pvi); get the bottom 10%
    # pvi_10_perc = np.percentile(list(score_sufficiency_pvi.values()), 10) -- this gives nan
    all_pvis = list(score_sufficiency_pvi.values())
    pvi_x_perc = sorted(all_pvis)[int(len(all_pvis) * perc)]

    print(f'pvi {perc*100} perc:', pvi_x_perc)  # -0.0650; -1.004490002287639e-08
    print(f'num pvis < {perc*100} perc:', sum([1 for pvi in all_pvis if pvi < pvi_x_perc]))

    return score_sufficiency_pvi, pvi_x_perc

# NOTE: fixed: example 44124 in uf (* Alabama - Montgomery) is missing quotation marks, which causing indexing issues
def load_uf_vinfo(perc=0.2):
    filename = '../checklist_out4/ultrafeedback_binarized_regular_vinfo_pvi_train.csv'  # checklist_out3 is t5, checklist_out4 is flan-t5-xl

    df = pd.read_csv(filename)
    vinfo_pvi = {}
    for i, row in df.iterrows():  # the index is for the unshuffed dataset
        vinfo_pvi[i] = float(row['PVI'])

    all_pvis = list(vinfo_pvi.values())  # 61135
    # pvi_x_perc = sorted(all_pvis)[int(len(all_pvis) * perc)]
    pvi_x_perc = 0.0  # remove everything with pvi below 0

    # get the number of pvis less than 10th percentile
    print(f'pvi {perc*100} perc:', pvi_x_perc)  # 10%: -0.513, 20%: -0.267
    print(f'num pvis < {perc*100} perc:', sum([1 for pvi in all_pvis if pvi < pvi_x_perc]))  # 10%: 6113, 20%: 12227

    # get the number of pvis < 0
    print('num pvis < 0:', sum([1 for pvi in all_pvis if pvi < 0]))  # 27188; 27164 for flan-t5

    return vinfo_pvi, pvi_x_perc


def load_uf_vinfo_chunk(low_high_perc_to_keep=(0.0, 0.2)):
    filename = '../checklist_out4/ultrafeedback_binarized_regular_vinfo_pvi_train.csv'  # checklist_out3 is t5, checklist_out4 is flan-t5-xl

    df = pd.read_csv(filename)
    vinfo_pvi = {}
    for i, row in df.iterrows():  # the index is for the unshuffed dataset
        vinfo_pvi[i] = float(row['PVI'])

    all_pvis = list(vinfo_pvi.values())  # 61135
    all_pvis = sorted(all_pvis)
    low_perc, high_perc = low_high_perc_to_keep
    pvi_low_perc = all_pvis[int(len(all_pvis) * low_perc)]
    pvi_high_perc = all_pvis[int(len(all_pvis) * high_perc)] if high_perc < 1.0 else all_pvis[-1]

    # get the number of pvis between the two percentiles
    print(f'pvi {low_perc*100} perc:', pvi_low_perc)
    print(f'pvi {high_perc*100} perc:', pvi_high_perc)
    print(f'num pvis between {low_perc*100} and {high_perc*100} perc:', sum([1 for pvi in all_pvis if pvi_low_perc <= pvi < pvi_high_perc]))

    return vinfo_pvi, (pvi_low_perc, pvi_high_perc)



if __name__ == "__main__":
    # noneng_feasibility_pvi, mean_pvi = load_uf_noneng_feasibility()
    # print(len(noneng_feasibility_pvi))

    score_sufficiency_pvi, pvi_10_perc = load_uf_score_sufficiency()
    print(len(score_sufficiency_pvi))

    # vinfo_pvi, pvi_10_perc = load_uf_vinfo()
    # print(len(vinfo_pvi))  # 61135