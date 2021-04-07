# change csv to json set
# coding: utf-8
from pathlib import Path
import numpy as np
from tqdm import tqdm
import pickle
import torch

cond, cnt = "test", 100000
data_path = Path(f"{cond}")
input_array = []
target_array = []
print(f"reading {cond} files")
for i in range(cnt):
    input_path = data_path / f"index/input_{i+1}.csv"
    target_path = data_path / f"accessibility/target_{i+1}.csv"

    input_array.append(torch.Tensor(np.loadtxt(input_path, delimiter=",", dtype=np.float).astype(np.int)))
    target_array.append(torch.Tensor(np.loadtxt(target_path, delimiter=",", dtype=np.float)))
print(f"saving to input_{cond}.pkl")
pickle.dump(torch.stack(input_array), open(f"input_{cond}.pkl", "wb"))
        
print(f"saving to target_{cond}.pkl")
pickle.dump(torch.stack(target_array), open(f"target_{cond}.pkl", "wb"))

