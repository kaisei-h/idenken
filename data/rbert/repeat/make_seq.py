#!python3
#coding: utf-8
import numpy as np
import sys

num = int(sys.argv[1])
max_length = int(sys.argv[2])
idx = int(sys.argv[3])

# 塩基割合 A:U:G:C:N
def prob():
        return np.random.dirichlet((1, 1, 1, 1, 0.1), 1)[0]


with open(f'sequence/seq{idx}.fa', 'w') as f:
        for i in range(num):
                # 各種長さを決めます
                min_weight = np.random.randint(1,4)
                length = np.random.randint(min_weight*100, max_length)
                repeat_len = np.random.randint(2,7)
                repeat_num = np.random.randint(2,7)
                roop_len = np.random.randint(0,length-repeat_len*repeat_num*2-1)
                front_len = np.random.randint(0,length-repeat_len*repeat_num*2-roop_len)
                back_len = length-repeat_len*repeat_num*2-roop_len-front_len


                # ランダム部分の配列を作ります
                front_seq = ''.join(np.random.choice(list('AUGCN'), size=front_len, p=prob()).tolist())
                back_seq = ''.join(np.random.choice(list('AUGCN'), size=back_len, p=prob()).tolist())
                roop_seq = ''.join(np.random.choice(list('AUGCN'), size=roop_len, p=prob()).tolist())

                # ステム部分の配列を作ります
                repeat_seq = ''.join(np.random.choice(list('AUGCN'), size=repeat_len, p=prob()).tolist()) * repeat_num
                complement_seq = repeat_seq.translate(str.maketrans({'A':"U",'U':"A",'G':"C",'C':"G"}))[::-1]

                # ステム部分を低確率で置換します
                repeat_seq = list(repeat_seq)
                complement_seq = list(complement_seq)
                # 置換する塩基を決める
                trans_s = np.random.binomial(1, np.random.beta(1, 30, len(repeat_seq)))
                trans_c = np.random.binomial(1, np.random.beta(1, 30, len(complement_seq)))
                # 置換の次の塩基は置換されやすくする
                for i in range(len(repeat_seq)):
                        if trans_s[i]==1:
                                repeat_seq[i] = np.random.choice(list('AUGCN'), size=1, p=prob())[0]
                                if i<len(repeat_seq)-1: trans_s[i+1] = np.random.binomial(1, np.random.beta(1, 1))
                        if trans_c[i]==1:
                                complement_seq[i] = np.random.choice(list('AUGCN'), size=1, p=prob())[0]
                                if i<len(complement_seq)-1: trans_c[i+1] = np.random.binomial(1, np.random.beta(1, 1))
                repeat_seq = ''.join(repeat_seq)
                complement_seq = ''.join(complement_seq)


                seq = front_seq + repeat_seq + roop_seq + complement_seq + back_seq
                f.writelines(f'>repeatlen{repeat_len}_repeatnum{repeat_num}_{repeat_seq}\n')
                f.writelines(seq)
                f.writelines('\n')