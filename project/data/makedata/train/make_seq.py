#!python3
#coding: utf-8
import numpy as np
from tqdm import tqdm
import sys

#いろんな長さ作るよう
length = sys.argv[2]
seq = [np.random.choice(list('ATGC')) for i in range(int(length))]
with open('random/sample_{}.txt'.format(int(sys.argv[1])), 'w') as f:
    f.writelines('>seq0\n')
    f.writelines(seq)
