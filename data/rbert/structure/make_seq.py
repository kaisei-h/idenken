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

                # 塩基割合 A:U:G:C:N
                prob = np.random.dirichlet((25, 25, 25, 25, 2), 1)[0]
                
                # ランダム部分の配列を作ります
                front_seq = ''.join(np.random.choice(list('AUGCN'), size=front_len, p=prob).tolist())
                back_seq = ''.join(np.random.choice(list('AUGCN'), size=back_len, p=prob).tolist())
                roop_seq = ''.join(np.random.choice(list('AUGCN'), size=roop_len, p=prob).tolist())
                
                # ステム部分の配列を作ります
                stem_seq = ''.join(np.random.choice(list('AUGCN'), size=stem_len, p=prob).tolist())
                complement_seq = stem_seq.translate(str.maketrans({'A':"U",'U':"A",'G':"C",'C':"G"}))[::-1]
                
                # ステム部分を低確率で置換します
                stem_seq = list(stem_seq)
                complement_seq = list(complement_seq)
                # 置換する塩基を決める
                trans_s = np.random.binomial(1, np.random.beta(1, 15, len(stem_seq)))
                trans_c = np.random.binomial(1, np.random.beta(1, 15, len(complement_seq)))
                # 置換の次の塩基は置換されやすくする
                for i in range(len(stem_seq)):
                        if trans_s[i]==1:
                                stem_seq[i] = np.random.choice(list('AUGCN'), size=1, p=prob)[0]
                                if i<len(stem_seq)-1: trans_s[i+1] = np.random.binomial(1, np.random.beta(2, 1))
                        if trans_c[i]==1:
                                complement_seq[i] = np.random.choice(list('AUGCN'), size=1, p=prob)[0]
                                if i<len(complement_seq)-1: trans_c[i+1] = np.random.binomial(1, np.random.beta(2, 1))
                stem_seq = ''.join(stem_seq)
                complement_seq = ''.join(complement_seq)


                seq = front_seq + stem_seq + roop_seq + complement_seq + back_seq

                f.writelines(f'>stemlen{stem_len}_transs{sum(trans_s)}_transc{sum(trans_c)}\n')
                f.writelines(seq)
                f.writelines('\n')