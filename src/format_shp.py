import os
import re
import json
import pandas as pd
from abc import ABC, abstractmethod
from typing import List
from datasets import load_dataset, Dataset

RESPONSE_TOKEN_1 = 'A'
RESPONSE_TOKEN_2 = 'B'
POST_TOKEN = "Context"
QUERY_TOKEN = "Question"

class SubSampler(ABC):
    """
    An abstract class for subsampling the training data.
    """
    @abstractmethod
    def subsample(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class RatioSampler(SubSampler):
    """
    A SubSampler for the training data, based on the score ratio.
    The number of examples per post is limited to prevent over-fitting.
    """
    def __init__(self, ratio_thresh: float, examples_per_post: int):
        self.ratio_thresh = ratio_thresh
        self.examples_per_post = examples_per_post

    def subsample(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[df["score_ratio"] >= self.ratio_thresh]
        df = df.groupby("post_id").apply(
            lambda x: x.sample(n=min(self.examples_per_post, len(x)))
        )
        df = df.sample(n=len(df))
        return df


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


def sample_shp_data_from_hf(ratio_thresh, examples_per_post, data_dir=None):
    if data_dir:
        train_data = load_dataset('stanfordnlp/shp', split='train', data_dir=data_dir)
        test_data = load_dataset('stanfordnlp/shp', split='validation', data_dir=data_dir)
    else:
        train_data = load_dataset('stanfordnlp/shp', split='train')
        test_data = load_dataset('stanfordnlp/shp', split='validation')

    # train_data = Dataset.from_dict(train_data[:1000])  #1000
    # test_data = Dataset.from_dict(test_data[:20])

    train_df = train_data.to_pandas()
    subsampler = RatioSampler(ratio_thresh, examples_per_post)
    train_df = subsampler.subsample(train_df)

    # convert back to dataset
    train_data = Dataset.from_pandas(train_df)
    return train_data, test_data


