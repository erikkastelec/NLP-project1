import os
import pandas as pd
from nltk.tokenize import word_tokenize
import numpy as np


def vocabulary(filename):
    with open(filename, encoding="utf8") as f:
        contents = f.read()
        tokens = word_tokenize(contents)
        unique, counts = np.unique(tokens, return_counts=True)
        return len(unique), len(tokens)
    

def avg_vocabulary(corpus, language):
    type_token_ratio = 0
    avg_V = 0
    filtered = corpus[corpus['language'] == language]
    for filename in filtered['filename']:
        V, N = vocabulary(filename)
        avg_V += V
        type_token_ratio += V/N

    type_token_ratio /= len(filtered)
    avg_V /= len(filtered)

    return type_token_ratio, avg_V


def avg_attribute_num(corpus, attr, language):
    counter = 0
    if type(attr) == str:
        filtered = corpus[corpus['language'] == language][attr]
    else:
        filtered = corpus[corpus['language'] == language].iloc[:, attr]
    for r in filtered:
        counter += len(r.split(', '))
    
    return counter / len(filtered)
