# coding: utf-8
import pickle
import numpy as np

arr = ['seq', 'target', 'kmer']
for i in arr:
    seq1 = pickle.load(open("pickled/{}1.pkl".format(i),"rb"))
    seq2 = pickle.load(open("pickled/{}2.pkl".format(i),"rb"))
    seq3 = pickle.load(open("pickled/{}3.pkl".format(i),"rb"))
    seq4 = pickle.load(open("pickled/{}4.pkl".format(i),"rb"))
    seq5 = pickle.load(open("pickled/{}5.pkl".format(i),"rb"))
    seq = np.concatenate([seq1, seq2, seq3, seq4, seq5])
    pickle.dump(seq, open("{}_stacked.pkl".format(i), "wb"), protocol=4)