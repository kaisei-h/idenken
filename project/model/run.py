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
input_val0 = pickle.load(open("../data/max_span100/input_test0.pkl","rb"))
target_val0 = pickle.load(open("../data/max_span100/target_test0.pkl","rb")) #256のみ
target_val0 = torch.flip(target_val0, dims=[1])
input_train0 = pickle.load(open("../data/max_span100/input_train0.pkl","rb"))
target_train0 = pickle.load(open("../data/max_span100/target_train0.pkl","rb")) #256以下
target_train0 = torch.flip(target_train0, dims=[1])
input_val1 = pickle.load(open("../data/max_span100/input_test0.pkl","rb"))
target_val1 = pickle.load(open("../data/max_span100/target_test0.pkl","rb")) #256のみ
target_val1 = torch.flip(target_val1, dims=[1])
input_train1 = pickle.load(open("../data/max_span100/input_train0.pkl","rb"))
target_train1 = pickle.load(open("../data/max_span100/target_train0.pkl","rb")) #256以下
target_train1 = torch.flip(target_train1, dims=[1])

input_all = torch.cat([input_train0, input_val0, input_train1, input_val1], dim=0)
target_all = torch.cat([target_train0, target_val0, target_train1, target_val1], dim=0)
dataset = model.Dataset(input_all, target_all)
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [1800000, 200000])

del input_val0, target_val0, input_train0, target_train0, input_val1, target_val1, input_train1, target_train1
gc.collect()


import math
def lambda_epoch(epoch):
    # スケジューラの設定
    max_epoch = 20
    return math.pow((1-epoch/max_epoch), 0.9)


batch_size = 64
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
dataloaders_dict = {'train': train_dataloader, 'val': val_dataloader}

for prm1 in [128, 256]:
    for prm2 in [1e-3, 1e-5, 1e-7]:
        datePrint(prm1, 'filter', prm2, 'lr')
        net = model.Fixed(num_layer=32, num_filters=prm1, flag=False).to(device)
        net.apply(model.weight_init) #重みの初期化適用

        optimizer = optim.Adam(net.parameters(), lr=prm2, weight_decay=1e-6, eps=1e-5)

        epochs = 20
        criterion = nn.MSELoss().to(device)
                
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch)
        train_loss_list, val_loss_list, data_all, target_all, output_all = mode.train(device, net, dataloaders_dict, criterion, optimizer, epochs)               
        
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
        plt.savefig(f'{val_loss_list[-1]:.2f}_{prm1}_{prm2}.png')
