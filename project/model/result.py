#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


def plot_result(y_true:np.array, y_est:np.array, lims=[-1, 15]) -> None:
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
    plt.show()
    
    
def learning_curve(train_loss_list, val_loss_list, epochs):
    plt.plot(range(epochs), val_loss_list, label='val')
    plt.plot(range(epochs), train_loss_list, label='train')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
    
    
def cor_hist(cor_list):
    plt.hist(cor_list)
    # plt.xlim(-1, 1)
    plt.xlabel('correlation coefficient')
    plt.ylabel('count')
    plt.show()
    print(np.average(cor_list))


def loss_hist(loss_list):
    plt.hist(loss_list)
    plt.xlabel('loss')
    plt.ylabel('count')
    plt.show()
    print(np.average(loss_list))

    
def scatter_minmax(cor_list, loss_list, target_all, output_all):
    plt.scatter(target_all[np.argmax(cor_list)], output_all[np.argmax(cor_list)], label=max(cor_list))
    plt.scatter(target_all[np.argmin(cor_list)], output_all[np.argmin(cor_list)], label=min(cor_list))
    plt.xlabel('target')
    plt.ylabel('output')
    plt.legend()
    plt.show()
    

def cal_indicators(target_all, output_all):
    cor_list = []
    loss_list = []
    for n in range(len(target_all)):
        mse = 0
        cor = np.corrcoef(target_all[n], output_all[n])
        cor_list.append(cor[0, 1])
        mse = ((target_all[n] - output_all[n])**2).mean(axis=0)
        loss_list.append(mse)

    return cor_list, loss_list


def sort_list(loss_list, cor_list):
    loss_sort = np.array(loss_list).argsort()
    cor_sort = np.array(cor_list).argsort()

    return loss_sort, cor_sort


def visible_one(target_all, output_all, data_all, loss_list, cor_list, idx=None):
    if (idx==None):
        ii = np.random.randint(0, len(target_all))
        print('random=',random)
    else:
        ii = idx
    plt.figure(figsize=(15, 7))
    plt.plot(np.array(range(len(target_all[0]))) , target_all[ii], label='target', color='b')
    plt.plot(np.array(range(len(target_all[0]))) , output_all[ii], label='output', color='r')
    plt.legend()
    plt.xlabel('base position')
    plt.ylabel('accessibility')
    plt.title(f'{ii}th array')
    print('loss', loss_list[ii])
    print('cor', cor_list[ii])
    plt.show()

    base_list = []
    for n in range(len(data_all[0])):
        if (data_all[ii][n]==1):
            base_list.append('A')
        elif (data_all[ii][n]==2):
            base_list.append('U')
        elif (data_all[ii][n]==3):
            base_list.append('G')
        elif (data_all[ii][n]==4):
            base_list.append('C')
        elif (data_all[ii][n]==0):
            continue
    print(''.join(base_list))


# def visible_minmax(target_all, output_all, cor_list, loss_list):
#     tmp = {'cor_max': np.argmax(cor_list), 
#            'cor_min': np.argmin(cor_list),
#            'loss_max': np.argmax(loss_list),
#            'loss_min': np.argmin(loss_list)} 
#     for k, v in tmp.items():
#         plt.figure(figsize=(15, 7))
#         # plt.bar(np.array(range(len(target_all[0])))-0.2 , target_all[v], label='target', color='b', width=0.4, align='center')
#         # plt.bar(np.array(range(len(target_all[0])))+0.2 , output_all[v], label='output', color='r', width=0.4, align='center')
#         plt.plot(np.array(range(len(target_all[0]))) , target_all[v], label='target', color='b')
#         plt.plot(np.array(range(len(target_all[0]))) , output_all[v], label='output', color='r')
#         plt.legend()
#         plt.title(k)
#         plt.show()


# def show_base(data_all, cor_list, loss_list):
#     tmp = {'cor_max': np.argmax(cor_list), 
#            'cor_min': np.argmin(cor_list),
#            'loss_max': np.argmax(loss_list),
#            'loss_min': np.argmin(loss_list)}
#     for k, v in tmp.items():
#         base_list = []
#         for i in range(len(data_all[0])):
#             if (data_all[v][i]==1):
#                 base_list.append('A')
#             elif (data_all[v][i]==2):
#                 base_list.append('U')
#             elif (data_all[v][i]==3):
#                 base_list.append('G')
#             elif (data_all[v][i]==4):
#                 base_list.append('C')
#             elif (data_all[v][i]==0):
#                 continue
#         print(k, ''.join(base_list))