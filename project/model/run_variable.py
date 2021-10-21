import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pickle
import gc
import re
from torchsummary import summary
# ここから自作
import model
import result
import mode
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def datePrint(*args, **kwargs):
    from datetime import datetime
    print(datetime.now().strftime('[%Y/%m/%d %H:%M:%S] '), end="")
    print(*args, **kwargs)



datePrint("loading pickle data")
input_val0 = pickle.load(open("../data/max_span100_512/input_val0.pkl","rb"))
target_val0 = pickle.load(open("../data/max_span100_512/target_val0.pkl","rb")) #512のみ
target_val0 = torch.flip(target_val0, dims=[1])
input_val1 = pickle.load(open("../data/max_span100_512/input_val1.pkl","rb"))
target_val1 = pickle.load(open("../data/max_span100_512/target_val1.pkl","rb")) #512のみ
target_val1 = torch.flip(target_val1, dims=[1])
input_val2 = pickle.load(open("../data/max_span100_512/input_val2.pkl","rb"))
target_val2 = pickle.load(open("../data/max_span100_512/target_val2.pkl","rb")) #512のみ
target_val2 = torch.flip(target_val2, dims=[1])
input_val3 = pickle.load(open("../data/max_span100_512/input_val3.pkl","rb"))
target_val3 = pickle.load(open("../data/max_span100_512/target_val3.pkl","rb")) #512のみ
target_val3 = torch.flip(target_val3, dims=[1])
# input_val4 = pickle.load(open("../data/max_span100_512/input_val4.pkl","rb"))
# target_val4 = pickle.load(open("../data/max_span100_512/target_val4.pkl","rb")) #512のみ
# target_val4 = torch.flip(target_val4, dims=[1])
# input_val5 = pickle.load(open("../data/max_span100_512/input_val5.pkl","rb"))
# target_val5 = pickle.load(open("../data/max_span100_512/target_val5.pkl","rb")) #512のみ
# target_val5 = torch.flip(target_val5, dims=[1])
# input_val6 = pickle.load(open("../data/max_span100_512/input_val6.pkl","rb"))
# target_val6 = pickle.load(open("../data/max_span100_512/target_val6.pkl","rb")) #512のみ
# target_val6 = torch.flip(target_val6, dims=[1])
input_pro0 = pickle.load(open("../data/max_span100_512/input_pro0.pkl","rb"))
target_pro0 = pickle.load(open("../data/max_span100_512/target_pro0.pkl","rb")) #精度が低かった塩基組成
target_pro0 = torch.flip(target_pro0, dims=[1])
input_pro1 = pickle.load(open("../data/max_span100_512/input_pro1.pkl","rb"))
target_pro1 = pickle.load(open("../data/max_span100_512/target_pro1.pkl","rb")) #精度が低かった塩基組成
target_pro1 = torch.flip(target_pro1, dims=[1])
input_pro2 = pickle.load(open("../data/max_span100_512/input_pro2.pkl","rb"))
target_pro2 = pickle.load(open("../data/max_span100_512/target_pro2.pkl","rb")) #精度が低かった塩基組成
target_pro2 = torch.flip(target_pro2, dims=[1])
input_pro3 = pickle.load(open("../data/max_span100_512/input_pro3.pkl","rb"))
target_pro3 = pickle.load(open("../data/max_span100_512/target_pro3.pkl","rb")) #精度が低かった塩基組成
target_pro3 = torch.flip(target_pro3, dims=[1])
# input_val7 = pickle.load(open("../data/max_span100_512/input_val7.pkl","rb"))
# target_val7 = pickle.load(open("../data/max_span100_512/target_val7.pkl","rb")) #512のみ
# target_val7 = torch.flip(target_val7, dims=[1])
# input_val8 = pickle.load(open("../data/max_span100_512/input_val8.pkl","rb"))
# target_val8 = pickle.load(open("../data/max_span100_512/target_val8.pkl","rb")) #512のみ
# target_val8 = torch.flip(target_val8, dims=[1])
# input_val9 = pickle.load(open("../data/max_span100_512/input_val9.pkl","rb"))
# target_val9 = pickle.load(open("../data/max_span100_512/target_val9.pkl","rb")) #512のみ
# target_val9 = torch.flip(target_val9, dims=[1])
# input_val10 = pickle.load(open("../data/max_span100_512/input_val10.pkl","rb"))
# target_val10 = pickle.load(open("../data/max_span100_512/target_val10.pkl","rb")) #512のみ
# target_val10 = torch.flip(target_val10, dims=[1])
# input_val11 = pickle.load(open("../data/max_span100_512/input_val11.pkl","rb"))
# target_val11 = pickle.load(open("../data/max_span100_512/target_val11.pkl","rb")) #512のみ
# target_val11 = torch.flip(target_val11, dims=[1])
# input_val12 = pickle.load(open("../data/max_span100_512/input_val12.pkl","rb"))
# target_val12 = pickle.load(open("../data/max_span100_512/target_val12.pkl","rb")) #512のみ
# target_val12 = torch.flip(target_val12, dims=[1])
# input_val13 = pickle.load(open("../data/max_span100_512/input_val13.pkl","rb"))
# target_val13 = pickle.load(open("../data/max_span100_512/target_val13.pkl","rb")) #512のみ
# target_val13 = torch.flip(target_val13, dims=[1])
# input_val14 = pickle.load(open("../data/max_span100_512/input_val14.pkl","rb"))
# target_val14 = pickle.load(open("../data/max_span100_512/target_val14.pkl","rb")) #512のみ
# target_val14 = torch.flip(target_val14, dims=[1])
# input_val15 = pickle.load(open("../data/max_span100_512/input_val15.pkl","rb"))
# target_val15 = pickle.load(open("../data/max_span100_512/target_val15.pkl","rb")) #512のみ
# target_val15 = torch.flip(target_val15, dims=[1])
# input_val16 = pickle.load(open("../data/max_span100_512/input_val16.pkl","rb"))
# target_val16 = pickle.load(open("../data/max_span100_512/target_val16.pkl","rb")) #512のみ
# target_val16 = torch.flip(target_val16, dims=[1])


input_all = torch.cat([input_val0,input_val1,input_val2,input_val3,input_pro0,input_pro1,input_pro2,input_pro3], dim=0)
target_all = torch.cat([target_val0,target_val1,target_val2,target_val3,target_pro0,target_pro1,target_pro2,target_pro3], dim=0)


kmer = []
for i in range(len(input_all)-2):
    kmer.append(int(str(int(input_all[i]))+str(int(input_all[i+1]))+str(int(input_all[i+2]))))
kmer = torch.Tensor(kmer)
print(kmer[:100])


split_rate = [3500000, 500000]
dataset = model.Dataset(input_all, target_all)
train_dataset, val_dataset = torch.utils.data.random_split(dataset, split_rate)
print(split_rate)

del input_val0,target_val0,input_val1,target_val1,input_val2,target_val2,input_val3,target_val3
del input_pro0,target_pro0,input_pro1,target_pro1,input_pro2,target_pro2,input_pro3,target_pro3
gc.collect()

import math
def lambda_epoch(epoch):
    # スケジューラの設定
    max_epoch = 20
    return math.pow((1-epoch/max_epoch), 0.9)


batch_size = 32
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
dataloaders_dict = {'train': train_dataloader, 'val': val_dataloader}

net = model.dilation11(num_layer=16, num_filters=128, kernel_sizes=5).to(device)
net.apply(model.weight_init) #重みの初期化適用
summary(net, (512,))
# print(net)

optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-8, eps=1e-5)
print(optimizer)
epochs = 20 
criterion = nn.MSELoss().to(device)

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch)
train_loss_list, val_loss_list, data_all, target_all, output_all = mode.train(device, net, dataloaders_dict, criterion, optimizer, epochs)               
torch.save(net.state_dict(), 'pro_data.pth')


y_true, y_est = np.array(target_all, dtype=object).reshape(-1), np.array(output_all, dtype=object).reshape(-1)
lims = [-1, 15]
fig,ax = plt.subplots(1,1,dpi=150,figsize=(5,5))
heatmap, xedges, yedges = np.histogram2d(y_true, y_est, bins=100,range=(lims,lims))
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
cset = ax.imshow(heatmap.T, extent=extent, origin='lower', norm=LogNorm(), cmap='rainbow')
ax.plot(lims, lims, ls="--", color="black", alpha=0.5, label="ideal")
plt.xlabel('target')
plt.ylabel('output')
ax.legend()

from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
fig.colorbar(cset, cax=cax).ax.set_title("count")
plt.savefig(f'pro_data_{val_loss_list[-1]:.2f}.png')
