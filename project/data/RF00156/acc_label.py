# coding: utf-8
import csv
import re
from tqdm import tqdm
import sys
import numpy as np


max_length = 252
with open('out_{}.txt'.format(sys.argv[1]), 'r') as f:
        next(f)
        acc = f.readlines()
        acc_list = []
        for i in range(len(acc)-1):
                acc_list.append(re.findall(',(.*);', acc[i])[0])

        acc_list = np.pad(acc_list, (max_length-len(acc_list), 0), 'constant')
        with open('target_{}.csv'.format(sys.argv[1]), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(acc_list)
