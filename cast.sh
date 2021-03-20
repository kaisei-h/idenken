#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -l s_vmem=32G
#$ -l mem_req=32G
#$ -e cast_data.log
#$ -o cast_data.log

source ~/.bashrc

cd /home/kaisei-h/project/data/RF00156
python3 cast_RF.py
