#!/bin/bash
#$ -cwd
#$ -m be
#$ -e ds.log
#$ -o ds.log
#$ -l s_vmem=64G
#$ -l mem_req=64G

source ~/.bashrc
echo start `date`
python3 stack.py
echo finish`date`
