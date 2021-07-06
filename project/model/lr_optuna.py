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
import optuna
# ここから自作
import model
import result
import mode
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


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
# input_val2 = pickle.load(open("../data/max_span100_512/input_val2.pkl","rb"))
# target_val2 = pickle.load(open("../data/max_span100_512/target_val2.pkl","rb")) #512のみ
# target_val2 = torch.flip(target_val2, dims=[1])
# input_val3 = pickle.load(open("../data/max_span100_512/input_val3.pkl","rb"))
# target_val3 = pickle.load(open("../data/max_span100_512/target_val3.pkl","rb")) #512のみ
# target_val3 = torch.flip(target_val3, dims=[1])
# input_val4 = pickle.load(open("../data/max_span100_512/input_val4.pkl","rb"))
# target_val4 = pickle.load(open("../data/max_span100_512/target_val4.pkl","rb")) #512のみ
# target_val4 = torch.flip(target_val4, dims=[1])
# input_val5 = pickle.load(open("../data/max_span100_512/input_val5.pkl","rb"))
# target_val5 = pickle.load(open("../data/max_span100_512/target_val5.pkl","rb")) #512のみ
# target_val5 = torch.flip(target_val5, dims=[1])
# input_val6 = pickle.load(open("../data/max_span100_512/input_val6.pkl","rb"))
# target_val6 = pickle.load(open("../data/max_span100_512/target_val6.pkl","rb")) #512のみ
# target_val6 = torch.flip(target_val6, dims=[1])


input_all = torch.cat([input_val0,input_val1], dim=0)
target_all = torch.cat([target_val0,target_val1], dim=0)


dataset = model.Dataset(input_all, target_all)
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [900000, 100000])

del input_val0,target_val0,input_val1,target_val1
gc.collect()

import math
def lambda_epoch(epoch):
    # スケジューラの設定
    max_epoch = 10
    return math.pow((1-epoch/max_epoch), 0.9)



def get_optimizer(trial, model):
    optimizer_names = ['Adam', 'MomentumSGD', 'rmsprop']
    optimizer_name = trial.suggest_categorical('optimizer', optimizer_names)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-9, 1e-3)
    if optimizer_name == optimizer_names[0]: 
        adam_lr = trial.suggest_loguniform('adam_lr', 1e-6, 1e-2)
        optimizer = optim.Adam(model.parameters(), lr=adam_lr, weight_decay=weight_decay)
    elif optimizer_name == optimizer_names[1]:
        momentum_sgd_lr = trial.suggest_loguniform('momentum_sgd_lr', 1e-6, 1e-2)
        optimizer = optim.SGD(model.parameters(), lr=momentum_sgd_lr, momentum=0.9, weight_decay=weight_decay)
    else:
        optimizer = optim.RMSprop(model.parameters())
        rmp_prop_lr = trial.suggest_loguniform('rms_prop_lr', 1e-6, 1e-2)
        optimizer = optim.Adam(model.parameters(), lr=rmp_prop_lr, weight_decay=weight_decay)

    return optimizer


def objective(trial):
    batch_size = 64
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    dataloaders_dict = {'train': train_dataloader, 'val': val_dataloader}
    
    net = model.dilation1(num_layer=16, num_filters=128, kernel_sizes=5).to(device)
    net.apply(model.weight_init) #重みの初期化適用
    optimizer = get_optimizer(trial, net)
    print(optimizer)
    criterion = nn.MSELoss().to(device)
    epochs = 10

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch)
    train_loss_list, val_loss_list, data_all, target_all, output_all = mode.train(device, net, dataloaders_dict, criterion, optimizer, epochs)

    return float(val_loss_list[-1])



trial = 50
study = optuna.create_study()
study.optimize(objective, n_trials=trial)

print(study.best_params)
print(study.best_value)