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

                repeat_len = np.random.randint(1,10) # 反復長さを1~10で決定
                max_repeat = int((length-1)//repeat_len) # 反復回数の最大値を取得
                repeat_num = np.random.randint(2,max_repeat) # 反復回数を決定

                front_len = np.random.randint(0,length-repeat_len*repeat_num) # 反復部分より手前の長さを決定
                back_len = length-repeat_len*repeat_num-front_len # #相補部分より後ろの長さを決定

                # ランダム部分の配列を作成
                front_seq = ''.join(np.random.choice(list('AUGCN'), size=front_len, p=prob()).tolist())
                back_seq = ''.join(np.random.choice(list('AUGCN'), size=back_len, p=prob()).tolist())

                # 反復部分を作成
                repeat_seq = ''.join(np.random.choice(list('AUGCN'), size=repeat_len, p=prob()).tolist()) * repeat_num

                # 反復部分を低確率で置換する
                # 置換のために一度リスト型に変換
                repeat_seq = list(repeat_seq)
                # 二項分布で低確率で置換する塩基位置を決定
                trans_s = np.random.binomial(1, np.random.beta(1, 30, len(repeat_seq)))
                # 置換していく
                for i in range(len(repeat_seq)):
                        if trans_s[i]==1:
                                repeat_seq[i] = np.random.choice(list('AUGCN'), size=1, p=prob())[0]
                                if i<len(repeat_seq)-1: trans_s[i+1] = np.random.binomial(1, np.random.beta(1, 1)) # 置換の次の塩基は置換されやすくする
                # 文字列に戻す
                repeat_seq = ''.join(repeat_seq)

                # 作ったパーツを足して1本の配列にする
                seq = front_seq + repeat_seq + back_seq

                f.writelines(f'>repeatlen{repeat_len}_repeatnum{repeat_num}_{repeat_seq}\n')
                f.writelines(seq)
                f.writelines('\n')