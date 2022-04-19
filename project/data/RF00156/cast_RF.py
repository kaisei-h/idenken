# change csv to json set
# coding: utf-8
from pathlib import Path
import numpy as np
from tqdm import tqdm
import pickle
import torch

cond, cnt = "RF00156", 1332
input_array = []
target_array = []
print(f"reading {cond} files")
for i in range(cnt):
    input_path = f"input_{i}.csv"
    target_path = f"target_{i}.csv"

    input_array.append(torch.Tensor(np.loadtxt(input_path, delimiter=",", dtype=np.float).astype(np.int)))
    target_array.append(torch.Tensor(np.loadtxt(target_path, delimiter=",", dtype=np.float)))
print(f"saving to input_{cond}.pkl")
pickle.dump(torch.stack(input_array), open(f"input_{cond}.pkl", "wb"))
        
print(f"saving to target_{cond}.pkl")
pickle.dump(torch.stack(target_array), open(f"target_{cond}.pkl", "wb"))

