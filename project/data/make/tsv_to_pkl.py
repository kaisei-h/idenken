import pandas as pd
import pickle
import collections
import numpy as np

def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab


vocab = load_vocab("forbert/vocab.txt")

df = pd.read_table('forbert/dev.tsv')
dev = df['sequence'].str.split(' ', expand=True).replace(vocab).drop(columns=508)
dev = np.array(dev.values).astype(np.float)
print(dev.shape)
pickle.dump(dev, open("forbert/dev.pkl", "wb"))

df = pd.read_table('forbert/train.tsv')
train = df['sequence'].str.split(' ', expand=True).replace(vocab).drop(columns=508)
train = np.array(train.values).astype(np.float)
print(train.shape)
pickle.dump(train, open("forbert/train.pkl", "wb"))