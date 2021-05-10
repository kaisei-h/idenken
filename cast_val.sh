#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -l s_vmem=16G
#$ -l mem_req=16G
#$ -e c_error.log
#$ -o c_out.log
#$ -t 1-5:1

source ~/.bashrc

cd /home/kaisei-h/project/data/makedata
python3 cast_val.py $SGE_TASK_ID