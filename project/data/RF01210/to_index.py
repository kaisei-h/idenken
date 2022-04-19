# coding: utf-8
import csv
import numpy as np
import sys


max_length = 256
with open('sample_{}.txt'.format(sys.argv[1]), 'r') as f:
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

        #いろんな長さ作るようpadding    
        seq_number=np.pad(seq_number, (0, max_length-len(seq_number)), 'constant')

        with open('input_{}.csv'.format(sys.argv[1]), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(seq_number)
