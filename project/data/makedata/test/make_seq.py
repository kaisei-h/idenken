#!python3
#coding: utf-8
import numpy as np
import sys

#固定データ作成用
seq = [np.random.choice(list('ATGC')) for i in range(int(256))]
#いろんな長さ作るよう
max_length = 256
#seq = [np.random.choice(list('AUGC')) for i in range(np.random.randint(16 ,max_length))]
with open('random/sample_{}.txt'.format(int(sys.argv[1])), 'w') as f:
    f.writelines('>seq0\n')
    f.writelines(seq)
