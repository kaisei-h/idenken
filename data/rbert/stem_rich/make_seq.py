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
                stem_len = np.random.randint(8,48)
                roop_len = np.random.randint(3,length-stem_len*2)
                front_len = np.random.randint(0,length-stem_len*2-roop_len)
                back_len = length-stem_len*2-roop_len-front_len

                # 塩基割合を決めます
                a = np.random.randint(20,50)
                u = np.random.randint(20,50)
                g = np.random.randint(20,50)
                c = np.random.randint(20,50)
                n = np.random.randint(0,5)
                total = a+u+g+c+n
                prob = [a/total, u/total, g/total, c/total, n/total]
                
                # ランダム部分の配列を作ります
                front_seq = ''.join(np.random.choice(list('AUGCN'), size=front_len, p=prob).tolist())
                back_seq = ''.join(np.random.choice(list('AUGCN'), size=back_len, p=prob).tolist())
                roop_seq = ''.join(np.random.choice(list('AUGCN'), size=roop_len, p=prob).tolist())
                
                # ステム部分の配列を作ります
                stem_seq = ''.join(np.random.choice(list('AUGCN'), size=stem_len, p=prob).tolist())
                complement_seq = stem_seq.translate(str.maketrans({'A':"U",'U':"A",'G':"C",'C':"G"}))[::-1]
                
                seq = front_seq + stem_seq + roop_seq + complement_seq + back_seq

                f.writelines(f'>stem_len{stem_len}\n')
                f.writelines(seq)
                f.writelines('\n')