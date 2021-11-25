#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -l s_vmem=128G
#$ -l mem_req=128G
#$ -e dc.log
#$ -o dc.log
#$ -t 1-1:1

source ~/.bashrc
name="dev"
echo start `date`
sum=100000
python3 cast.py $sum $SGE_TASK_ID $name
echo finish`date`