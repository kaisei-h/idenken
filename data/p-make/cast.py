# change csv to json set
# coding: utf-8
from pathlib import Path
import numpy as np
import pickle
import sys

cnt = int(sys.argv[1])
seq_array = []
target_array = []

for i in range(cnt):
    x = i + (int(sys.argv[2])-1)*cnt +1
    seq_path = f"index/idx{x}.csv"
    target_path = f"basepairprob/bpp{x}.csv"

    try:
        seq_array.append(np.loadtxt(seq_path, delimiter=",", dtype=str))
        target_array.append(np.loadtxt(target_path, delimiter=",", dtype=np.float))    
    except Exception as e:
        print(e)
        print(f"basepairprob/bpp{x}.csv")
    

print(f"saving to seq{sys.argv[3]}.pkl")
pickle.dump(np.stack(seq_array), open(f"pickled/seq_{sys.argv[2]}.pkl", "wb"))

print(f"saving to target_{sys.argv[3]}.pkl")
pickle.dump(np.stack(target_array), open(f"pickled/target_{sys.argv[2]}.pkl", "wb"))