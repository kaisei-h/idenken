# change csv to json set
# coding: utf-8
from pathlib import Path
import numpy as np
import pickle
import sys

cnt = int(sys.argv[1])
# seq_array = []
kmer_array = ''
target_array = []

for i in range(cnt):
    x = i + (int(sys.argv[2])-1)*cnt +1
    # seq_path = f"sequence/inp{x}.csv"
    kmer_path = f"k-mer/k_inp{x}.txt"
    target_path = f"accessibility/acc{x}.csv"
    # seq_array.append(np.loadtxt(seq_path, delimiter=",", dtype=str))
    f = open(kmer_path, 'r')
    kmer_array += f.read() + '\t1\n'
    target_array.append(np.loadtxt(target_path, delimiter=",", dtype=np.float))

# print(f"saving to seq{sys.argv[2]}.pkl")
# pickle.dump(np.stack(seq_array), open(f"pickled/seq{sys.argv[2]}.pkl", "wb"))

print(f"saving to {sys.argv[3]}.tsv")
with open(f"pickled/{sys.argv[3]}.tsv", 'w') as f:
    f.writelines('sequence\tlabel\n')
    f.writelines(kmer_array)

print(f"saving to target_{sys.argv[3]}.pkl")
pickle.dump(np.stack(target_array), open(f"pickled/target_{sys.argv[3]}.pkl", "wb"))

