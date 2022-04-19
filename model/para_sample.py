import pandas as pd

frame = []
with open('sample.txt') as f:
    line = f.readlines()
    for l in line:
        if l[0] == '#':
            continue
        else:
            frame.append(list(map(float, (l.rstrip('\n').split('\t')[1:]))))
index, columns, value = zip(*frame)

df = pd.DataFrame(index=list(map(str, range(4, 201))), columns=list(map(str, range(5,513))))
for i, c, v in zip(index, columns, value):
    idx, col = str(int(c-i)), str(int(c))
    df[col][idx] = v

df.to_csv('sample.csv')