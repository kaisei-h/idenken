#!python3
#coding: utf-8
import numpy as np
import sys
import csv

length = int(sys.argv[2])
prob = [0.25, 0.25, 0.25, 0.25]
seq = np.random.choice(list('ATGC'), size=length, p=prob)

with open('random/seq{}.txt'.format(int(sys.argv[1])), 'w') as f:
# with open('random/seq_test.txt', 'w') as f:
    f.writelines('>seq0\n')
    f.writelines(seq)

with open('sequence/inp{}.csv'.format(int(sys.argv[1])), 'w') as f:
# with open('sequence/inp_test.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(seq)

k = 5
kmer = ''
for i in range(len(seq)-(k-1)):
        kmer += (seq[i]+seq[i+1]+seq[i+2]+seq[i+3]+seq[i+4]+' ') #k変更時に+seq[i+k-1]を書き加える
with open('k-mer/k_inp{}.txt'.format(int(sys.argv[1])), 'w') as f:
    # f.writelines('sequence\tlabel\n')
    f.writelines(kmer)