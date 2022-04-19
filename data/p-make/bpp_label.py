# coding: utf-8
import sys
import pandas as pd
import numpy as np
import csv

frame = []
with open(f'p-out/out{sys.argv[1]}.txt') as f:
    for l in f:
        if '#' in l:
            continue
        else:
            frame.append(list(map(float, (l.rstrip('\n').split('\t')[1:]))))

try:
    index, columns, value = zip(*frame)
except Exception as e:
    print(e)
    print(f'p-out/out{sys.argv[1]}.txt')

df = pd.DataFrame(index=list(map(str, range(1, 513))), columns=list(map(str, range(1,513))))
for i, c, v in zip(index, columns, value):
    idx, col = str(int(i)), str(int(c))
    df[col][idx] = v

df = df.fillna(0)
df = df.to_numpy()

# with open(f'basepairprob/bpp{sys.argv[1]}.csv', 'w') as f:
#     writer = csv.writer(f)
#     writer.writerow(df)

np.savetxt(f'basepairprob/bpp{sys.argv[1]}.csv', df, delimiter=',')