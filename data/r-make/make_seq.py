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
# n = np.random.randint(0,10)
n = 0
total = a+t+g+c+n

prob = [a/total, t/total, g/total, c/total, n/total]
seq = np.random.choice(list('AUGCN'), size=length, p=prob)

with open('random/seq{}.fa'.format(int(sys.argv[1])), 'w') as f:
# with open('random/seq_test.txt', 'w') as f:
    f.writelines('>randomseq\n')
    f.writelines(seq)

with open('sequence/inp{}.csv'.format(int(sys.argv[1])), 'w') as f:
# with open('sequence/inp_test.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(seq)

seq_number = []
for i in range(len(seq)):
        if seq[i]=='A':
                seq_number.append(2)
        elif seq[i]=='U':
                seq_number.append(3)
        elif seq[i]=='G':
                seq_number.append(4)
        elif seq[i]=='C':
                seq_number.append(5)
        elif seq[i]=='N':
                seq_number.append(1)

with open('index/idx{}.csv'.format(sys.argv[1]), 'w') as f:
    writer = csv.writer(f)
    writer.writerow(seq_number)

# k = 5
# kmer = ''
# for i in range(len(seq)-(k-1)):
#         kmer += (seq[i]+seq[i+1]+seq[i+2]+seq[i+3]+seq[i+4]+' ') #k変更時に+seq[i+k-1]を書き加える
# with open('k-mer/k_inp{}.txt'.format(int(sys.argv[1])), 'w') as f:
#     # f.writelines('sequence\tlabel\n')
#     f.writelines(kmer)