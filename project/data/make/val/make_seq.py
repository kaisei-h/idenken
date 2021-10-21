#!python3
#coding: utf-8
import numpy as np
import sys
import csv

length = int(sys.argv[2])
# prob = [0.24, 0.25, 0.26, 0.25] #lossが悪かった塩基組成
prob = [0.25, 0.25, 0.25, 0.25]
seq = np.random.choice(list('ATGC'), size=length, p=prob)

with open('random/sample_{}.txt'.format(int(sys.argv[1])), 'w') as f:
    f.writelines('>seq0\n')
    f.writelines(seq)

# k = 2
# kmer = []
# for i in range(len(seq)-(k-1)):
#     kmer.append(seq[i]+seq[i+1])
# kmer = np.array(kmer)

# with open('kmer/k_inp_{}.csv'.format(int(sys.argv[1])), 'w') as f:
#         writer = csv.writer(f)
#         writer.writerow(kmer)

