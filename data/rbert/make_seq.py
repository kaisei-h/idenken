#!python3
#coding: utf-8
import numpy as np
import sys
import csv

num = int(sys.argv[1])
max_length = int(sys.argv[2])
idx = int(sys.argv[3])



with open(f'sequence/seq{idx}.fa', 'w') as f:
        for i in range(num):
                # 長さを決めます
                min_weight = np.random.randint(1,4)
                length = np.random.randint(min_weight*100, max_length)

                a = np.random.randint(20,50)
                u = np.random.randint(20,50)
                g = np.random.randint(20,50)
                c = np.random.randint(20,50)
                n = np.random.randint(0,5)
                total = a+u+g+c+n

                prob = [a/total, u/total, g/total, c/total, n/total]
                seq = np.random.choice(list('AUGCN'), size=length, p=prob)
                A = np.count_nonzero(seq=='A')
                U = np.count_nonzero(seq=='U')
                G = np.count_nonzero(seq=='G')
                C = np.count_nonzero(seq=='C')
                N = np.count_nonzero(seq=='N')
                f.writelines(f'>A{A}_U{U}_G{G}_C{C}_N{N}\n')
                f.writelines(seq)
                f.writelines('\n')