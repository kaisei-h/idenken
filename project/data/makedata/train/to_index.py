# coding: utf-8
import csv
import numpy as np
from tqdm import tqdm
import sys

length = sys.argv[2]
with open('random/sample_{}.txt'.format(sys.argv[1]), 'r') as f:
        next(f)
        seq = f.read()
        seq =list(seq)
        seq_number = []
        for i in range(len(seq)):
                if seq[i]=='A':
                        seq_number.append(1)
                elif seq[i]=='T':
                        seq_number.append(2)
                elif seq[i]=='G':
                        seq_number.append(3)
                elif seq[i]=='C':
                        seq_number.append(4)

        with open('index/input_{}.csv'.format(sys.argv[1]), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(seq_number)