#!/bin/bash
#$ -cwd
#$ -m be
#$ -l gpu
#$ -l cuda=2
#$ -l s_vmem=96G
#$ -l mem_req=96G
#$ -l d_rt=120:00:00
#$ -l s_rt=120:00:00

source ~/.bashrc

cd /home/kaisei-h/project/model
python3 w20.py