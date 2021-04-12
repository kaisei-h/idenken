# one-hotに変換する配列
# coding: utf-8
import csv
import numpy as np
from tqdm import tqdm
import sys

with open('random/sample_{}.txt'.format(sys.argv[1]), 'r') as f:
	next(f)
	seq = f.read()
	seq =list(seq)
	seq_number = []
	for i in range(len(seq)):
		if seq[i]=='A':
			seq_number.append(0)
		elif seq[i]=='U':
			seq_number.append(1)
		elif seq[i]=='G':
			seq_number.append(2)
		elif seq[i]=='C':
			seq_number.append(3)

	one_hot = np.identity(4)[seq_number]
	one_hot = one_hot.T

	with open('one-hot/input_{}.csv'.format(sys.argv[1]), 'w') as f:
		writer = csv.writer(f)
		writer.writerows(one_hot)
