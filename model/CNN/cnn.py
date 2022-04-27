import random
import copy
from Bio import SeqIO
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import numpy as np
import itertools
import gc
import csv

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import result
import model

from datetime import datetime, timedelta, timezone
JST = timezone(timedelta(hours=+9), 'JST')
dt_now = datetime.now(JST)
dt_now = dt_now.strftime('%Y%m%d-%H%M%S')
print(dt_now)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'torch.cuda.device_count()={torch.cuda.device_count()}')

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
        convered_seq = [kmer_dict[i] for i in s] + [0]*(max_length - len(s))
        seq_num.append(convered_seq)
    return seq_num

def make_dict(k=3):
    # seq to num 
    l = ["A", "U", "G", "C"]
    kmer_list = [''.join(v) for v in list(itertools.product(l, repeat=k))]
    kmer_list.insert(0, "MASK")
    dic = {kmer: i+1 for i,kmer in enumerate(kmer_list)}
    return dic

class AccDataset(torch.utils.data.Dataset):
    def __init__(self, low_seq, seq_len, accessibility):
        self.data_num = len(low_seq)
        self.low_seq = low_seq
        self.seq_len = seq_len
        self.accessibility = accessibility
        
    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        out_low_seq = self.low_seq[idx]
        out_seq_len = self.seq_len[idx]
        out_accessibility = self.accessibility[idx]

        return out_low_seq, out_seq_len, out_accessibility

def make_dl(seq_data_path, acc_data_path):
    data_sets = seq_data_path
    seqs = []
    for i, data_set in enumerate(data_sets):
        for record in SeqIO.parse(data_set, "fasta"):
            record = record[::-1] #reverseします
            seq = str(record.seq).upper()
            seqs.append(seq)

    data_sets = acc_data_path
    accessibility = []
    for i, data_set in enumerate(data_sets):
        with open(data_set) as f:
            reader = csv.reader(f)
            for l in reader:
                accessibility.append(l)

    accessibility = torch.tensor(np.array(accessibility, dtype=np.float32))
    seqs_len = np.tile(np.array([len(i) for i in seqs]), 1)
    k = 1
    kmer_seqs = kmer(seqs, k)
    masked_seq, low_seq = mask(kmer_seqs, rate = 0.02, mag = 1)
    kmer_dict = make_dict(k)
    swap_kmer_dict = {v: k for k, v in kmer_dict.items()}
    low_seq = torch.tensor(np.array(convert(low_seq, kmer_dict, max_length)))


    ds_ACC = AccDataset(low_seq, seqs_len, accessibility)
    dl_ACC = torch.utils.data.DataLoader(ds_ACC, batch_size, shuffle=True)
    return dl_ACC

def model_device(device, model):
    print("device: ", device)
    model.to(device)
    model = torch.nn.DataParallel(model) # make parallel
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return model


class ForAcc(nn.Module):
    def __init__(self):
        super(ForAcc, self).__init__()

        self.embedding = nn.Embedding(6, 120)        
        
        self.convs = nn.ModuleList()
        self.convs.append(model.conv1DBatchNormMish(in_channels=120, out_channels=120,
                                                    kernel_size=3, padding=1, dilation=1))
        self.convs.append(model.scSE(channels=120))
        for i in range(30):
            self.convs.append(model.conv1DBatchNormMish(in_channels=120, out_channels=240,
                                                    kernel_size=7, padding=3*9, dilation=9))
            self.convs.append(model.conv1DBatchNormMish(in_channels=240, out_channels=480,
                                                    kernel_size=7, padding=3*5, dilation=5))
            self.convs.append(model.conv1DBatchNormMish(in_channels=480, out_channels=120,
                                                    kernel_size=7, padding=3*1, dilation=1))
        
        self.convs.append(model.scSE(channels=120))
        self.convs.append(model.conv1DBatchNorm(in_channels=120, out_channels=1, kernel_size=6))
        

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = torch.transpose(x, 1, 2)
        for i, l in enumerate(self.convs):
            x = l(x)        
        x = x.view(x.shape[0], -1)

        return x
    
comment = '100k_data'

max_length ,batch_size = 440, 100
seq_path = '../../data/rbert/sequence/'
acc_path = '../../data/rbert/accessibility/'

train_seq_paths = []
train_acc_paths = []
for n in range(1):
    train_seq_paths.append([f'{seq_path}seq{n*i+1}.fa' for i in range(10)])
    train_acc_paths.append([f'{acc_path}acc{n*i+1}.csv' for i in range(10)])

val_seq_path = [f'{seq_path}seq{i+1}.fa' for i in range(90,100)]
val_acc_path = [f'{acc_path}acc{i+1}.csv' for i in range(90,100)]

adam_lr = 3e-4
model = ForAcc()
model = model_device(device, model)
optimizer = optim.RAdam([{'params': model.parameters(), 'lr': adam_lr}])
criterion = nn.MSELoss().to(device)
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
val_dl_ACC = make_dl(val_seq_path, val_acc_path)

for epoch in range(epochs):
    print(f'Epoch {epoch+1}/{epochs}')
    
    for phase in ['train', 'val']:
        start = time.time()
    
        if (epoch==0) and (phase=='train'):
            continue
        
        if phase == 'train':
            model.train()
            avg_losses = []
            epoch_loss = 0
            for train_seq_path, train_acc_path in zip(train_seq_paths, train_acc_paths):
                train_dl_ACC = make_dl(train_seq_path, train_acc_path)
                for batch in train_dl_ACC:
                    low_seq, _, accessibility = batch
                    data = low_seq.to(device, non_blocking=False)
                    target = accessibility.to(device, non_blocking=False)
                    optimizer.zero_grad()
                    if data.size()[0] == 1:
                        continue
                    with torch.cuda.amp.autocast():
                        with torch.set_grad_enabled:
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
                low_seq, _, accessibility = batch
                data = low_seq.to(device, non_blocking=False)
                target = accessibility.to(device, non_blocking=False)

                optimizer.zero_grad()
                if data.size()[0] == 1:
                    continue
                with torch.cuda.amp.autocast():
                    output = model(data)
                    if (epoch+1)==epochs:
                        data_all.append(data.cpu().numpy())
                        target_all.append(target.cpu().numpy())
                        output_all.append(output.cpu().numpy())
                    
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
result.plot_result(target_all.reshape(-1), output_all.reshape(-1), mode='save')

train_time = sum(train_time_list) / len(train_time_list)
val_time = sum(val_time_list) / len(val_time_list)



cor_list, loss_list = result.cal_indicators(target.cpu().numpy(), output.cpu().numpy())


dt_now = datetime.now(JST)
dt_now = dt_now.strftime('%Y%m%d-%H%M%S')
print(dt_now)
with open(f'../jupyter-log/{dt_now}.log', 'w') as f:
    f.writelines(f'comment: {comment}\n')
    f.writelines(f'loss: {loss_list[-1]} \n')
    f.writelines(f'cor: {sum(cor_list)/len(cor_list)} \n')
    f.writelines(f'train_time: {train_time} \n')
    f.writelines(f'val_time: {val_time} \n')
    f.writelines(f'train_loss: {train_loss_list} \n')
    f.writelines(f'val_loss: {val_loss_list} \n')
    f.writelines(f'criterion: {criterion} \n')
    f.writelines(f'optimizer: {optimizer} \n')
    f.writelines(f'model: {model} \n')

