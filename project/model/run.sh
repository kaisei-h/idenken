#!/bin/bash
#$ -cwd
#$ -m be
#$ -l gpu
#$ -l cuda=1
#$ -l s_vmem=96G
#$ -l mem_req=96G
#$ -l d_rt=120:00:00
#$ -l s_rt=120:00:00

source ~/.bashrc
python3 run.py
