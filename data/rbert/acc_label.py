# coding: utf-8
import csv
import re
import sys

num = int(sys.argv[1])
length = int(sys.argv[2])
idx = int(sys.argv[3])

with open(f'w50/output/out{idx}.txt', 'r') as fr:
        with open(f'w50/accessibility/acc{idx}.csv', 'w') as fw:
                for i, l in enumerate(fr):
                        if (l[0]=='>'):
                                acc_list = []
                        elif (l[0]=='\n'):
                                continue
                        elif (l[0]=='0'):
                                acc_list.append(re.findall(',(.*);', l)[0])
                                writer = csv.writer(fw)
                                writer.writerow(acc_list)
                        else:
                                acc_list.append(re.findall(',(.*);', l)[0])