#!python3
#coding: utf-8
import numpy as np
from tqdm import tqdm
import sys

length = int(sys.argv[2])
prob = [0.245, 0.255, 0.26, 0.24] #corが悪かった塩基組成
seq = np.random.choice(list('ATGC'), size=length, p=prob)
with open('random/sample_{}.txt'.format(int(sys.argv[1])), 'w') as f:
    f.writelines('>seq0\n')
    f.writelines(seq)
