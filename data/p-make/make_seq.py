#!python3
#coding: utf-8
import numpy as np
import sys
import csv

length = int(sys.argv[2])

a = np.random.randint(30,50)
t = np.random.randint(30,50)
g = np.random.randint(30,50)
c = np.random.randint(30,50)
n = np.random.randint(0,10)
total = a+t+g+c+n

prob = [a/total, t/total, g/total, c/total, n/total]
seq = np.random.choice(list('ATGCN'), size=length, p=prob)

with open('random/seq{}.txt'.format(int(sys.argv[1])), 'w') as f:
    f.writelines('>randomseq\n')
    f.writelines(seq)

# with open('sequence/inp{}.csv'.format(int(sys.argv[1])), 'w') as f:
#     writer = csv.writer(f)
#     writer.writerow(seq)

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
        elif seq[i]=='N':
                seq_number.append(5)

with open('index/idx{}.csv'.format(sys.argv[1]), 'w') as f:
    writer = csv.writer(f)
    writer.writerow(seq_number)