#!/bin/bash
#$ -cwd
#$ -m be
#$ -l s_vmem=32G
#$ -l mem_req=32G

source ~/.bashrc
python3 cat_train.py
python3 cat_val.py