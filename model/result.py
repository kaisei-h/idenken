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
    plt.xlim(-1, 1)
    plt.xlabel('correlation coefficient')
    plt.ylabel('count')
    plt.show()
    print(np.average(cor_list))

def loss_hist(loss_list):
    plt.hist(loss_list)
    plt.xlim(xmin=0)
    plt.xlabel('loss')
    plt.ylabel('count')
    plt.show()
    print(np.average(loss_list))    

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

def remake_bad(target_all, output_all, data_all, loss_sort, cor_sort, length=1000):
    loss_target = []
    loss_output = []
    loss_data = []
    count_A, count_U, count_G, count_C = 0, 0, 0, 0
    for i in range(length):
        loss_target.append(target_all[loss_sort[-i-1]])
        loss_output.append(output_all[loss_sort[-i-1]])
        loss_data.append(data_all[loss_sort[-i-1]])
    
        count_A += np.count_nonzero(data_all[loss_sort[-i-1]]==1)
        count_U += np.count_nonzero(data_all[loss_sort[-i-1]]==2)
        count_G += np.count_nonzero(data_all[loss_sort[-i-1]]==3)
        count_C += np.count_nonzero(data_all[loss_sort[-i-1]]==4)
    total = count_A + count_U + count_G + count_C
    print(f'A:{count_A/total:.3f}, U:{count_U/total:.3f}, G:{count_G/total:.3f}, C:{count_C/total:.3f}')

    cor_target = []
    cor_output = []
    cor_data = []
    count_A, count_U, count_G, count_C = 0, 0, 0, 0
    for i in range(length):
        cor_target.append(target_all[cor_sort[i]])
        cor_output.append(output_all[cor_sort[i]])
        cor_data.append(data_all[cor_sort[i]])

        count_A += np.count_nonzero(data_all[cor_sort[i]]==1)
        count_U += np.count_nonzero(data_all[cor_sort[i]]==2)
        count_G += np.count_nonzero(data_all[cor_sort[i]]==3)
        count_C += np.count_nonzero(data_all[cor_sort[i]]==4)

    total = count_A + count_U + count_G + count_C
    print(f'A:{count_A/total:.3f}, U:{count_U/total:.3f}, G:{count_G/total:.3f}, C:{count_C/total:.3f}')

    return loss_target, loss_output, loss_data, cor_target, cor_output, cor_data

def remake_good(target_all, output_all, data_all, loss_sort, cor_sort, length=1000):
    loss_target = []
    loss_output = []
    loss_data = []
    count_A, count_U, count_G, count_C = 0, 0, 0, 0
    for i in range(length):
        loss_target.append(target_all[loss_sort[i]])
        loss_output.append(output_all[loss_sort[i]])
        loss_data.append(data_all[loss_sort[i]])
    
        count_A += np.count_nonzero(data_all[loss_sort[i]]==1)
        count_U += np.count_nonzero(data_all[loss_sort[i]]==2)
        count_G += np.count_nonzero(data_all[loss_sort[i]]==3)
        count_C += np.count_nonzero(data_all[loss_sort[i]]==4)
    total = count_A + count_U + count_G + count_C
    print(f'A:{count_A/total:.3f}, U:{count_U/total:.3f}, G:{count_G/total:.3f}, C:{count_C/total:.3f}')

    cor_target = []
    cor_output = []
    cor_data = []
    count_A, count_U, count_G, count_C = 0, 0, 0, 0
    for i in range(length):
        cor_target.append(target_all[cor_sort[-i-1]])
        cor_output.append(output_all[cor_sort[-i-1]])
        cor_data.append(data_all[cor_sort[-i-1]])

        count_A += np.count_nonzero(data_all[cor_sort[-i-1]]==1)
        count_U += np.count_nonzero(data_all[cor_sort[-i-1]]==2)
        count_G += np.count_nonzero(data_all[cor_sort[-i-1]]==3)
        count_C += np.count_nonzero(data_all[cor_sort[-i-1]]==4)

    total = count_A + count_U + count_G + count_C
    print(f'A:{count_A/total:.3f}, U:{count_U/total:.3f}, G:{count_G/total:.3f}, C:{count_C/total:.3f}')

    return loss_target, loss_output, loss_data, cor_target, cor_output, cor_data


def count_diff(data_all):
    diff_list = []
    for i in data_all:
        count_A, count_U, count_G, count_C = 0, 0, 0, 0
        count_A += np.count_nonzero(i==1)
        count_U += np.count_nonzero(i==2)
        count_G += np.count_nonzero(i==3)
        count_C += np.count_nonzero(i==4)
        diff = abs(count_A-len(i)/4) + abs(count_U-len(i)/4) + abs(count_G-len(i)/4) + abs(count_C-len(i)/4)
        diff_list.append(diff)

    return diff_list

def heat_scatter(x:np.array, y:np.array) -> None:
    fig,ax = plt.subplots(1,1,dpi=150,figsize=(5,5))
    counts, xedges, yedges, Image = ax.hist2d(x, y, bins=50, norm=LogNorm(),)
    fig.colorbar(Image, ax=ax)
    plt.xlabel('diff')
    plt.ylabel('accuracy')
    plt.show()


def visible_one(target_all, output_all, data_all, loss_list, cor_list, idx=None):
    if (idx==None):
        ii = np.random.randint(0, len(target_all))
        print('random=',ii)
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

    count_A = np.count_nonzero(data_all[ii]==1)
    count_U = np.count_nonzero(data_all[ii]==2)
    count_G = np.count_nonzero(data_all[ii]==3)
    count_C = np.count_nonzero(data_all[ii]==4)
    total = count_A + count_U + count_G + count_C
    print(f'A:{count_A/total:.3f}, U:{count_U/total:.3f}, G:{count_G/total:.3f}, C:{count_C/total:.3f}')

    # base_list = []
    # for n in range(len(data_all[0])):
    #     if (data_all[ii][n]==1):
    #         base_list.append('A')
    #     elif (data_all[ii][n]==2):
    #         base_list.append('U')
    #     elif (data_all[ii][n]==3):
    #         base_list.append('G')
    #     elif (data_all[ii][n]==4):
    #         base_list.append('C')
    #     elif (data_all[ii][n]==0):
    #         continue
    
    # print(''.join(base_list))