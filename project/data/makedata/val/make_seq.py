#!python3
#coding: utf-8
import numpy as np
import sys

#固定データ作成用
length = int(sys.argv[2])
prob = [0.24, 0.25, 0.26, 0.25] #lossが悪かった塩基組成
seq = [np.random.choice(list('ATGC'), p=prob) for i in range(length)]
with open('random/sample_{}.txt'.format(int(sys.argv[1])), 'w') as f:
    f.writelines('>seq0\n')
    f.writelines(seq)
