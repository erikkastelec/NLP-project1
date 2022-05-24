import os
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
import numpy as np


def vocabulary(filename):
    with open(filename, encoding="utf8") as f:
        contents = f.read()
        tokens = word_tokenize(contents)
        unique, counts = np.unique(tokens, return_counts=True)
        return len(unique), len(tokens)
    

def avg_vocabulary(corpus, language='both'):
    type_token_ratio = 0
    avg_V = 0
    avg_N = 0
    if language == 'both':
        filtered = corpus
    else:
        filtered = corpus[corpus['language'] == language]
    for filename in filtered['filename']:
        V, N = vocabulary(filename)
        avg_V += V
        avg_N += N
        type_token_ratio += V/N

    return type_token_ratio / len(filtered), avg_V / len(filtered), avg_N / len(filtered)


def avg_attribute_num(corpus, attr, language='both'):
    counter = 0
    cor = corpus
    if language != 'both':
        cor = corpus[corpus['language'] == language]
    
    if type(attr) == str:
        filtered = cor[attr]
    else:
        filtered = cor.iloc[:, attr]

    for r in filtered:
        counter += len(r.split(', '))
    
    return counter / len(filtered)

def avg_word_in_sentences(corpus, language='both'):
    if language == 'both':
        filtered = corpus
    else:
        filtered = corpus[corpus['language'] == language]
    st_ratio = 0
    for filename in filtered['filename']:
        with open(filename, encoding="utf8") as f:
            contents = f.read()
            sentences = sent_tokenize(contents)
            tokens = word_tokenize(contents)
            st_ratio += len(tokens) / len(sentences)
    return st_ratio / len(filtered)
