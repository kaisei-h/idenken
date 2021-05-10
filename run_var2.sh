#!/bin/bash
#$ -cwd
#$ -m be
#$ -l gpu
#$ -l cuda=1
#$ -l s_vmem=96G
#$ -l mem_req=96G
#$ -e r_v_error2.log
#$ -o r_v_out2.log
#$ -l d_rt=200:00:00
#$ -l s_rt=200:00:00

source ~/.bashrc

cd /home/kaisei-h/project/model
python3 run_var2.py