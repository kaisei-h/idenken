#!/bin/bash
#$ -cwd
#$ -m be
#$ -l gpu
#$ -l cuda=1
#$ -l s_vmem=96G
#$ -l mem_req=96G
#$ -e r_error.log
#$ -o r_out.log
#$ -l d_rt=120:00:00
#$ -l s_rt=120:00:00

source ~/.bashrc
cd /home/kaisei-h/project/model
python3 run_variable.py
