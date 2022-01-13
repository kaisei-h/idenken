#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -l s_vmem=16G
#$ -l mem_req=16G
#$ -e dc.log
#$ -o dc.log
#$ -t 1-5:1

source ~/.bashrc
name="train"
echo start `date`
sum=100000
python3 cast.py $sum $SGE_TASK_ID $name
# python3 tsv_to_pkl.py
echo finish`date`