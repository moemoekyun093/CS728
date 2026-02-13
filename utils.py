import numpy as np
from collections import defaultdict
from tqdm import tqdm
import json


def load_vocab(path):
    with open(path, "r") as f:
        vocab = json.load(f)
    return vocab


def build_cooccurrence_matrix(corpus, vocab, window_size):
    """
    corpus: list of tokenized documents
    vocab: dict mapping word -> index
    """
    cooc = defaultdict(float)

    for tokens in tqdm(corpus):
        for i, word in enumerate(tokens):
            if word not in vocab:
                continue
            w_i = vocab[word]

            start = max(0, i - window_size)
            end = min(len(tokens), i + window_size + 1)

            for j in range(start, end):
                if i == j:
                    continue
                context_word = tokens[j]
                if context_word not in vocab:
                    continue

                w_j = vocab[context_word]
                distance = abs(i - j)
                cooc[(w_i, w_j)] += 1.0 / distance

    return cooc
