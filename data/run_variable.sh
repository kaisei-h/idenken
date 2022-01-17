#!/bin/bash
#$ -cwd
#$ -m be
#$ -l gpu
#$ -l cuda=2
#$ -l s_vmem=96G
#$ -l mem_req=96G
# #$ -e r_v_error.log
# #$ -o r_v_out.log
#$ -l d_rt=200:00:00
#$ -l s_rt=200:00:00

source ~/.bashrc

cd /home/kaisei-h/project/model
python3 run_variable.py