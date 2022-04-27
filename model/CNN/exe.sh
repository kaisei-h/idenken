#!/bin/bash
#$ -cwd
#$ -V
#$ -m be
#$ -l gpu
#$ -l cuda=1
#$ -l s_vmem=96G
#$ -l mem_req=96G
#$ -l d_rt=240:00:00
#$ -l s_rt=240:00:00

source ~/.bashrc

cd /home/kaisei-h/model/CNN
python3 cnn.py