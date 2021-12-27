#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -l s_vmem=96G
#$ -l mem_req=96G
#$ -e dc.log
#$ -o dc.log
#$ -t 1-1:1

source ~/.bashrc
name="train"
echo start `date`
sum=100000
python3 cast.py $sum $SGE_TASK_ID $name
# python3 tsv_to_pkl.py
echo finish`date`