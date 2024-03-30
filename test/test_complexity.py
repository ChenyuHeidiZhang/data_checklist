import nltk
from nltk.corpus import brown, stopwords
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
import string
import numpy as np
import functools
import matplotlib.pyplot as plt

from transformers import AutoTokenizer
from datasets import load_dataset

from src.data_transforms import InputTransforms
from src.format_shp import sample_shp_data_from_hf

def test_basic():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('brown')  # Download the Brown Corpus

    stop_words = set(stopwords.words('english'))

    # Create a frequency distribution of words in the Brown Corpus
    freq_dist = FreqDist(brown.words())

    # sent = 'This is an example sentence'
    sent = 'This is an example sentence extraordinarily complex. Show encyl.'
    for word in word_tokenize(sent):
        if not word.lower() in stop_words and word not in string.punctuation:
            f = freq_dist.freq(word.lower())
            if f == 0:
                print(word, 'does not exist')
            else:
                print(word, np.log(f))

def test_hh():
    tokenizer = AutoTokenizer.from_pretrained('t5-base')
    transform_cls = InputTransforms(tokenizer=tokenizer)
    # input_data = load_dataset(Anthropic/hh-rlhf, split='test', data_dir='harmless-base')
    train_data, eval_data = sample_shp_data_from_hf(2, 5)
    print(len(train_data))

    input_transforms = functools.partial(
        transform_cls.transform_wrapper, input_transform_names=['shp_word_complexity'])
    train_data = train_data.map(input_transforms)

    # get distribution over complexity
    all_complexity = []
    for ex in train_data:
        inp = ex['input']
        complexity_A, complexity_B = inp.split('. Word complexity B: ')
        complexity_A = float(complexity_A.split('Word complexity A: ')[1])
        complexity_B = float(complexity_B[:-1])
        all_complexity.append(complexity_A)
        all_complexity.append(complexity_B)

    sorted_arr = np.sort(all_complexity)
    # print each quarter of the sorted array
    print(sorted_arr[len(sorted_arr)//4])  # 9.219206987507793
    print(sorted_arr[len(sorted_arr)//2])  # 9.5415876320688
    print(sorted_arr[3*len(sorted_arr)//4])  # 9.888243203800775
    # print(all_complexity)
    plt.hist(all_complexity, bins=20)
    plt.savefig('complexity.png')


if __name__ == '__main__':
    # test_basic()
    test_hh()
