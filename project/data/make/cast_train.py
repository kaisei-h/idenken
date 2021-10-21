# change csv to json set
# coding: utf-8
from pathlib import Path
import numpy as np
from tqdm import tqdm
import pickle
import torch
import sys
def datePrint(*args, **kwargs):
    from datetime import datetime
    print(datetime.now().strftime('[%Y/%m/%d %H:%M:%S] '), end="")
    print(*args, **kwargs)

cond, cnt = "train", 100000
data_path = Path(f"{cond}")
input_array = []
target_array = []
datePrint(f"reading {cond} files")
for i in range(cnt):
    x = i + (int(sys.argv[1])-1)*100000
    input_path = data_path / f"index/input_{x+1}.csv"
    # input_path = data_path / f"kmer/k_inp_{x+1}.csv"
    target_path = data_path / f"accessibility/target_{x+1}.csv"
    input_array.append(torch.Tensor(np.loadtxt(input_path, delimiter=",", dtype=np.float).astype(np.int)))
    target_array.append(torch.Tensor(np.loadtxt(target_path, delimiter=",", dtype=np.float)))
    if (i==cnt/2):
        datePrint("やっと半分だよ")
print(f"saving to input_train{sys.argv[1]}.pkl")
pickle.dump(torch.stack(input_array), open(f"input_train{sys.argv[1]}.pkl", "wb"))
        
print(f"saving to target_train{sys.argv[1]}.pkl")
pickle.dump(torch.stack(target_array), open(f"target_train{sys.argv[1]}.pkl", "wb"))

