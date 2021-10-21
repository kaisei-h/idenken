# coding: utf-8
import csv
import re
from tqdm import tqdm
import sys
import numpy as np


length = sys.argv[2]
with open('random_out/out_{}.txt'.format(sys.argv[1]), 'r') as f:
        next(f)
        acc = f.readlines()
        acc_list = []
        for i in range(len(acc)-1):
                acc_list.append(re.findall(',(.*);', acc[i])[0])

        with open('accessibility/target_{}.csv'.format(sys.argv[1]), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(acc_list)
