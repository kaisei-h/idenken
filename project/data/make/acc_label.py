# coding: utf-8
import csv
import re
import sys

length = sys.argv[2]
with open('r-out/out{}.txt'.format(sys.argv[1]), 'r') as f:
        next(f)
        acc = f.readlines()
        acc_list = []
        for i in range(len(acc)-1):
                acc_list.append(re.findall(',(.*);', acc[i])[0])

        with open('accessibility/acc{}.csv'.format(sys.argv[1]), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(acc_list)