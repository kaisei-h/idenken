#!python3
#coding: utf-8
import numpy as np
import sys

num = int(sys.argv[1])
max_length = int(sys.argv[2])
idx = int(sys.argv[3])



with open(f'sequence/seq{idx}.fa', 'w') as f:
        for i in range(num):
                # 各種長さを決めます
                min_weight = np.random.randint(1,4)
                length = np.random.randint(min_weight*100, max_length)

                # 塩基割合 A:U:G:C:N
                prob = np.random.dirichlet((1, 1, 1, 1, 0.1), 1)[0]

                seq = ''.join(np.random.choice(list('AUGCN'), size=length, p=prob).tolist())

                f.writelines(f'>prob{prob}\n')
                f.writelines(seq)
                f.writelines('\n')