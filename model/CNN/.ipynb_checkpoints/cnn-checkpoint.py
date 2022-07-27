import random
import copy
import time
import os
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
import model
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
    dl_ACC = torch.utils.data.DataLoader(ds_ACC, batch_size, num_workers=2, shuffle=False)

    return dl_ACC, flag, division


def model_device(device, model):
    print("device: ", device)
    model.to(device)
    model = torch.nn.DataParallel(model, device_ids=[0, 1]) # make parallel
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return model

def objective(trained_para=torch.load('bertrna.pth')):
    config.hidden_size = config.num_attention_heads * config.multiple
    # config.adam_lr = 3e-4
    # config.num_hidden_layers = 2
    
    # model = BertModel(config)
    # model = RBERT_CNN(model)
    
    # model = CNN()
    # model = DCNN()
    model = TransCNN()
    
    model = model_device(device, model)

    # if args.pretraining:
    #     model.load_state_dict(torch.load(args.pretraining))
    model.load_state_dict(trained_para, strict=False)
        
    # optimizer = optim.RAdam([{'params': model.parameters(), 'lr': config.adam_lr}])
    optimizer = optim.AdamW([{'params': model.parameters(), 'lr': config.adam_lr}])
    return model, optimizer, config

class RBERT_CNN(nn.Module):
    def __init__(self, net_bert):
        super(RBERT_CNN, self).__init__()
        self.bert = net_bert
        self.convs = nn.ModuleList()
        self.convs.append(model.conv1DBatchNormRelu(in_channels=120, out_channels=120, kernel_size=3, padding=1, dilation=1))
        self.convs.append(model.scSE(channels=120))
        for i in range(10):
            self.convs.append(model.conv1DBatchNormRelu(in_channels=120, out_channels=120, kernel_size=7, padding=3*5, dilation=5))
        self.convs.append(model.scSE(channels=120))
        self.convs.append(model.conv1DBatchNormRelu(in_channels=120, out_channels=480, kernel_size=1))
        self.convs.append(model.conv1DBatchNorm(in_channels=480, out_channels=1, kernel_size=9, padding=4))
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
        self.convs.append(model.conv1DBatchNormRelu(in_channels=120, out_channels=120, kernel_size=3, padding=1, dilation=1))
        self.convs.append(model.scSE(channels=120))
        for i in range(10):
            self.convs.append(model.conv1DBatchNormRelu(in_channels=120, out_channels=120, kernel_size=7, padding=3*5, dilation=5))
        self.convs.append(model.scSE(channels=120))
        self.convs.append(model.conv1DBatchNormRelu(in_channels=120, out_channels=480, kernel_size=1))
        self.convs.append(model.conv1DBatchNorm(in_channels=480, out_channels=1, kernel_size=9, padding=4))        
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
        self.convs.append(model.conv1DBatchNormRelu(in_channels=120, out_channels=120, kernel_size=3, padding=1, dilation=1))
        self.convs.append(model.scSE(channels=120))
        for i in range(100):
            self.convs.append(model.conv1DBatchNormRelu(in_channels=120, out_channels=120, kernel_size=7, padding=3*5, dilation=5))
        self.convs.append(model.scSE(channels=120))
        self.convs.append(model.conv1DBatchNormRelu(in_channels=120, out_channels=480, kernel_size=1))
        self.convs.append(model.conv1DBatchNorm(in_channels=480, out_channels=1, kernel_size=9, padding=4))        
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = torch.transpose(x, 1, 2)
        for i, l in enumerate(self.convs):
            x = l(x)        
        return x.view(x.shape[0], -1)

class TransCNN(nn.Module):
    def __init__(self):
        super(TransCNN, self).__init__()
        self.embedding = nn.Embedding(6, 120)
        self.trans_layer = nn.TransformerEncoderLayer(d_model=120, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.trans_layer, num_layers=6)
        self.convs = nn.ModuleList()
        self.convs.append(model.conv1DBatchNormRelu(in_channels=120, out_channels=120, kernel_size=3, padding=1, dilation=1))
        self.convs.append(model.scSE(channels=120))
        for i in range(10):
            self.convs.append(model.conv1DBatchNormRelu(in_channels=120, out_channels=120, kernel_size=7, padding=3*5, dilation=5))
        self.convs.append(model.scSE(channels=120))
        self.convs.append(model.conv1DBatchNormRelu(in_channels=120, out_channels=480, kernel_size=1))
        self.convs.append(model.conv1DBatchNorm(in_channels=480, out_channels=1, kernel_size=9, padding=4))        
    def forward(self, input_ids):
        x = self.transformer(self.embedding(input_ids))
        x = torch.transpose(x, 1, 2)
        for i, l in enumerate(self.convs):
            x = l(x)        
        return x.view(x.shape[0], -1)
    

comment = '100k_data'

max_length ,batch_size = 440, 200
seq_path = '../../data/rbert/structure/sequence/'
acc_path = '../../data/rbert/structure/accessibility/'

train_seq_paths = []
train_acc_paths = []
for n in range(2):
    train_seq_paths.append([f'{seq_path}seq{n*10+i+1}.fa' for i in range(2)])
    train_acc_paths.append([f'{acc_path}acc{n*10+i+1}.csv' for i in range(2)])

val_seq_path = [f'{seq_path}seq{i+1}.fa' for i in range(999,1000)]
val_acc_path = [f'{acc_path}acc{i+1}.csv' for i in range(999,1000)]


criterion = nn.MSELoss().to(device)
config = get_config(file_path = "./RNA_bert_config.json")
model, optimizer, config = objective()
epochs = 10

model.to(device)
torch.backends.cudnn.benchmark = True
scaler = torch.cuda.amp.GradScaler()

train_loss_list = []
val_loss_list = []
train_time_list = []
val_time_list = []
data_all = []
target_all = []
output_all = []
val_dl_ACC, flag, division = make_dl(val_seq_path, val_acc_path)

for epoch in range(epochs):
    print(f'Epoch {epoch+1}/{epochs}')
    
    for phase in ['train', 'val']:
        start = time.time()
    
        if (epoch==0) and (phase=='train'):
            continue
        
        if phase == 'train':
            model.train()
            avg_losses = []
            for train_seq_path, train_acc_path in zip(train_seq_paths, train_acc_paths):
                train_dl_ACC, _, _ = make_dl(train_seq_path, train_acc_path)
                epoch_loss = 0
                for batch in train_dl_ACC:
                    low_seq, accessibility = batch
                    data = low_seq.to(device, non_blocking=False)
                    target = accessibility.to(device, non_blocking=False)
                    optimizer.zero_grad()
                    if data.size()[0] == 1:
                        continue
                    with torch.cuda.amp.autocast():
                        with torch.set_grad_enabled(phase=='train'):
                            output = model(data)
                            loss = criterion(output, target)                            
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                            epoch_loss += loss.item() * data.size(0)
                            
                avg_losses.append(epoch_loss / len(train_dl_ACC.dataset))
            avg_loss = sum(avg_losses) / len(avg_losses)

        else:
            model.eval()            
            epoch_loss = 0
            for batch in val_dl_ACC:
                low_seq, accessibility = batch
                data = low_seq.to(device, non_blocking=False)
                target = accessibility.to(device, non_blocking=False)

                optimizer.zero_grad()
                if data.size()[0] == 1:
                    continue
                with torch.cuda.amp.autocast():
                    output = model(data)
                    if (epoch+1)==epochs:
                        data_all.append(data.cpu().detach().numpy())
                        target_all.append(target.cpu().detach().numpy())
                        output_all.append(output.cpu().detach().numpy())
                    
                    loss = criterion(output, target)
                    epoch_loss += loss.item() * data.size(0)         
            avg_loss = epoch_loss / len(val_dl_ACC.dataset)
        
        finish = time.time()
        print(f'{phase} Loss:{avg_loss:.4f} Timer:{finish - start:.4f}')

        if phase=='val':
            val_time_list.append(finish - start)
            val_loss_list.append(avg_loss)
            if avg_loss<0.1:
                break
        elif phase=='train':
            train_time_list.append(finish - start)
            train_loss_list.append(avg_loss)
            
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
               }, f'mypth/{comment}.pth')

data_all = np.concatenate(data_all)
target_all = np.concatenate(target_all)
output_all = np.concatenate(output_all)
cor_list, loss_list = result.cal_indicators(target_all, output_all)

train_time = sum(train_time_list) / len(train_time_list)
val_time = sum(val_time_list) / len(val_time_list)

pad_removed_target, pad_removed_output = result.remove_padding(target_all, output_all)
true_loss = ((np.array(pad_removed_target) - np.array(pad_removed_output))**2).mean(axis=0)
result.plot_result(pad_removed_target, pad_removed_output, mode='save')

dt_now = datetime.now(JST)
dt_now = dt_now.strftime('%Y%m%d-%H%M%S')
print(dt_now)
with open(f'mylog/{dt_now}.log', 'w') as f:
    f.writelines(f'comment: {comment}\n')
    f.writelines(f'true_loss: {true_loss} \n')
    f.writelines(f'cor: {sum(cor_list)/len(cor_list)} \n')
    f.writelines(f'train_time: {train_time} \n')
    f.writelines(f'val_time: {val_time} \n')
    f.writelines(f'train_loss: {train_loss_list} \n')
    f.writelines(f'val_loss: {val_loss_list} \n')
    f.writelines(f'config: {config} \n')
    f.writelines(f'criterion: {criterion} \n')
    f.writelines(f'optimizer: {optimizer} \n')
    f.writelines(f'params_update: {params_update} \n')
    f.writelines(f'model: {model} \n')