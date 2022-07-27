import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1"

import random
import copy
import time
import gc
import csv
import sys
import itertools
import more_itertools
import glob
import json
from datetime import datetime, timedelta, timezone

sys.path.append('..')
import result
import mymodel
import mode

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Bio import SeqIO
from RNABERT.utils.bert import BertModel, get_config

JST = timezone(timedelta(hours=+9), 'JST')
dt_now = datetime.now(JST)
dt_now = dt_now.strftime('%Y%m%d-%H%M%S')
print(dt_now)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.device_count()


def kmer(seqs, k=1):
    #塩基文字列をk-mer文字列リストに変換
    kmer_seqs = []
    for seq in seqs:
        kmer_seq = []
        for i in range(len(seq)):
            if i <= len(seq)-k:
                kmer_seq.append(seq[i:i+k])
        kmer_seqs.append(kmer_seq)
    return kmer_seqs

def mask(seqs, rate = 0.2, mag = 1):
    # 与えられた文字列リストに対してmask。rateはmaskの割合,magは生成回数/1配列
    seq = []
    masked_seq = []
    label = []
    for i in range(mag):
        seqs2 = copy.deepcopy(seqs)
        for s in seqs2:
            label.append(copy.copy(s))
            mask_num = int(len(s)*rate)
            all_change_index = np.array(random.sample(range(len(s)), mask_num))
            mask_index, base_change_index = np.split(all_change_index, [int(all_change_index.size * 0.90)])
#             index = list(np.sort(random.sample(range(len(s)), mask_num)))
            for i in list(mask_index):
                s[i] = "MASK"
            for i in list(base_change_index):
                s[i] = random.sample(('A', 'U', 'G', 'C'), 1)[0] 
            masked_seq.append(s)
    return masked_seq, label

def convert(seqs, kmer_dict, max_length):
    # 文字列リストを数字に変換
    seq_num = [] 
    if not max_length:
        max_length = max([len(i) for i in seqs])
    for s in seqs:
        # NやWやYとかよくわからん塩基をMASKにしてしまおうと
        convered_seq = [kmer_dict[i] if i in kmer_dict.keys() else 1 for i in s] + [0]*(max_length - len(s))
        seq_num.append(convered_seq)
    return seq_num

class AccDataset(torch.utils.data.Dataset):
    def __init__(self, low_seq, accessibility):
        self.data_num = len(low_seq)
        self.low_seq = low_seq
        # self.seq_len = seq_len
        self.accessibility = accessibility
        
    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        out_low_seq = self.low_seq[idx]
        # out_seq_len = self.seq_len[idx]
        out_accessibility = self.accessibility[idx]

        return out_low_seq, out_accessibility


def make_dl(seq_data_path, acc_data_path):
    flag = False
    division = 1
    max_length = 440
    
    data_sets = seq_data_path
    seqs = []
    for i, data_set in enumerate(data_sets):
        for record in SeqIO.parse(data_set, "fasta"):
            record = record[::-1] #reverseします
            seq = str(record.seq).upper()
            seqs.append(seq)
    seqs_len = np.tile(np.array([len(i) for i in seqs]), 1)

    if (len(seqs_len)>250000 or max(seqs_len)>25000 or len(seqs_len)*max(seqs_len)>100000000):
        return 'skip', f'seq_num{len(seqs_len)}', f'max_len{max(seqs_len)}'

    if max(seqs_len)>max_length:
        start = time.time()
        flag = True
        division += (max(seqs_len)-110) // 330
        max_length += division * 330
        finish = time.time()

    k = 1
    kmer_seqs = kmer(seqs, k)
    masked_seq, low_seq = mask(kmer_seqs, rate = 0.02, mag = 1)
    kmer_dict = {'MASK': 1, 'A': 2, 'U': 3, 'T': 3, 'G': 4, 'C': 5}
    low_seq = torch.tensor(np.array(convert(low_seq, kmer_dict, max_length)))

    if flag:
        splited_seq = []
        for i in low_seq:
            splited_seq.append(list(more_itertools.windowed(i ,440 ,step = 330)))
        low_seq = torch.tensor(splited_seq)
        num_seq, division, length = low_seq.shape
        low_seq = low_seq.view(-1, length)

    data_sets = acc_data_path
    accessibility = []
    for i, data_set in enumerate(data_sets):
        with open(data_set) as f:
            reader = csv.reader(f)
            for l in reader:
                pad_acc = l + ['-1']*(max_length - len(l))
                accessibility.append(pad_acc)
    accessibility = torch.tensor(np.array(accessibility, dtype=np.float32))

    if flag:
        start = time.time()
        splited_acc = []
        for i in accessibility:
            splited_acc.append(list(more_itertools.windowed(i ,440 ,step = 330)))
        accessibility = torch.tensor(splited_acc)
        accessibility = accessibility.view(-1, length)

        finish = time.time()

    ds_ACC = AccDataset(low_seq, accessibility)
    dl_ACC = torch.utils.data.DataLoader(ds_ACC, batch_size, num_workers=2, shuffle=True)

    return dl_ACC, flag, division


def model_device(device, model):
    print("device: ", device)
    model.to(device)
    model = torch.nn.DataParallel(model, device_ids=[0, 1]) # make parallel
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return model

class RBERT(nn.Module):
    def __init__(self, net_bert):
        super(RBERT, self).__init__()
        self.bert = net_bert
        self.convs = nn.ModuleList()
        self.convs.append(mymodel.conv1DBatchNormRelu(in_channels=120, out_channels=120, kernel_size=3, padding=1, dilation=1))
        self.convs.append(mymodel.scSE(channels=120))
        for i in range(10):
            self.convs.append(mymodel.conv1DBatchNormRelu(in_channels=120, out_channels=120, kernel_size=7, padding=3*5, dilation=5))
        self.convs.append(mymodel.scSE(channels=120))
        self.convs.append(mymodel.conv1DBatchNormRelu(in_channels=120, out_channels=480, kernel_size=1))
        self.convs.append(mymodel.conv1DBatchNorm(in_channels=480, out_channels=1, kernel_size=9, padding=4))
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, attention_show_flg=False):
        encoded_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False, attention_show_flg=False)
        x = encoded_layers
        x = torch.transpose(x, 1, 2)
        for i, l in enumerate(self.convs):
            x = l(x)
        return x.view(x.shape[0], -1)

class NBERT(nn.Module):
    def __init__(self, net_bert):
        super(NBERT, self).__init__()
        self.bert = net_bert
        self.convs = nn.ModuleList()
        self.convs.append(mymodel.conv1DBatchNormRelu(in_channels=120, out_channels=120, kernel_size=3, padding=1, dilation=1))
        self.convs.append(mymodel.scSE(channels=120))
        for i in range(10):
            self.convs.append(mymodel.conv1DBatchNormRelu(in_channels=120, out_channels=120, kernel_size=7, padding=3*5, dilation=5))
        self.convs.append(mymodel.scSE(channels=120))
        self.convs.append(mymodel.conv1DBatchNormRelu(in_channels=120, out_channels=480, kernel_size=1))
        self.convs.append(mymodel.conv1DBatchNorm(in_channels=480, out_channels=1, kernel_size=9, padding=4))
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, attention_show_flg=False):
        encoded_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False, attention_show_flg=False)
        x = encoded_layers
        x = torch.transpose(x, 1, 2)
        for i, l in enumerate(self.convs):
            x = l(x)
        return x.view(x.shape[0], -1)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(6, 120)        
        self.convs = nn.ModuleList()
        self.convs.append(mymodel.conv1DBatchNormRelu(in_channels=120, out_channels=120, kernel_size=3, padding=1, dilation=1))
        self.convs.append(mymodel.scSE(channels=120))
        for i in range(10):
            self.convs.append(mymodel.conv1DBatchNormRelu(in_channels=120, out_channels=120, kernel_size=7, padding=3*5, dilation=5))
        self.convs.append(mymodel.scSE(channels=120))
        self.convs.append(mymodel.conv1DBatchNormRelu(in_channels=120, out_channels=480, kernel_size=1))
        self.convs.append(mymodel.conv1DBatchNorm(in_channels=480, out_channels=1, kernel_size=9, padding=4))        
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = torch.transpose(x, 1, 2)
        for i, l in enumerate(self.convs):
            x = l(x)        
        return x.view(x.shape[0], -1)

class DCNN(nn.Module):
    def __init__(self):
        super(DCNN, self).__init__()
        self.embedding = nn.Embedding(6, 120)        
        self.convs = nn.ModuleList()
        self.convs.append(mymodel.conv1DBatchNormRelu(in_channels=120, out_channels=120, kernel_size=3, padding=1, dilation=1))
        self.convs.append(mymodel.scSE(channels=120))
        for i in range(50):
            self.convs.append(mymodel.conv1DBatchNormRelu(in_channels=120, out_channels=120, kernel_size=7, padding=3*5, dilation=5))
        self.convs.append(mymodel.scSE(channels=120))
        self.convs.append(mymodel.conv1DBatchNormRelu(in_channels=120, out_channels=480, kernel_size=1))
        self.convs.append(mymodel.conv1DBatchNorm(in_channels=480, out_channels=1, kernel_size=9, padding=4))        
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = torch.transpose(x, 1, 2)
        for i, l in enumerate(self.convs):
            x = l(x)        
        return x.view(x.shape[0], -1)
    


max_length, batch_size = 440, 16
criterion = nn.MSELoss().to(device)
config = get_config(file_path = "../RNABERT/RNA_bert_config.json")
config.hidden_size = config.num_attention_heads * config.multiple    

for ds in ['mouse', 'Rfam', 'human']:
    if ds=='mouse':
        seq_paths = sorted(glob.glob(f'../../data/GENCODE/mouse/seq*.fa'))
        acc_paths = sorted(glob.glob(f'../../data/GENCODE/mouse/acc*.csv'))
    elif ds=='Rfam':
        seq_paths = sorted(glob.glob(f'../../data/Rfam/RF*.fa'))
        acc_paths = sorted(glob.glob(f'../../data/Rfam/acc*.csv'))
    elif ds=='human':
        seq_paths = sorted(glob.glob(f'../../data/GENCODE/human/seq*.fa'))
        acc_paths = sorted(glob.glob(f'../../data/GENCODE/human/acc*.csv'))

    start = time.time()

    target_all = []
    output_all = []

    model = BertModel(config)
    model = RBERT(model)
    model = model_device(device, model)
    model.load_state_dict(torch.load('path/repeat_RBERT.pth')['model_state_dict'])

    for seq, acc in zip(seq_paths, acc_paths):
        print(seq, acc)
        dl, flag, division = make_dl([seq], [acc])

        if dl=="skip":
            print(f'{dl}, {flag}, {division}')
            continue

        data, target, output, loss, test_time = mode.test(device, model, dl, criterion)

        # 長い配列を復元するコード
        if flag:
            for i in range(division):
                if i==0:
                    low_tar = target[i::division, :-55]
                    low_out = output[i::division, :-55]
                elif i==division-1:
                    target = np.concatenate([low_tar, target[i::division, 55:]], axis=1)
                    output = np.concatenate([low_out, output[i::division, 55:]], axis=1)
                else:
                    low_tar = np.concatenate([low_tar, target[i::division, 55:-55]], axis=1)
                    low_out = np.concatenate([low_out, output[i::division, 55:-55]], axis=1)
                    
        target_all.append(target)
        output_all.append(output)
        
    finish = time.time()
    print(f'全データ_makedl含む_{finish-start}秒')

    target_rem, output_rem = result.remove_padding(target, output)
    print(f'True Loss: {((np.array(target_rem) - np.array(output_rem))**2).mean(axis=0)}')
    result.plot_result(target_rem, output_rem, lims=[-1.5, 23], mode='save')

    JST = timezone(timedelta(hours=+9), 'JST')
    dt_now = datetime.now(JST)
    dt_now = dt_now.strftime('%Y%m%d-%H%M%S')
    print(dt_now)