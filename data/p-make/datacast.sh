#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -l s_vmem=32G
#$ -l mem_req=32G
#$ -e dc.log
#$ -o dc.log
#$ -t 1-5:1

source ~/.bashrc
name="50000"
echo start `date`
sum=10000
python3 cast.py $sum $SGE_TASK_ID $name
echo finish`date`