#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -V
#$ -t 1-50000:1000
#$ -e label.log
#$ -o label.log


source ~/.bashrc
echo start`date`

cd /home/kaisei-h/data/p-make
for ((i = $SGE_TASK_ID; i < $SGE_TASK_ID+1000; i++)){
	python3 bpp_label.py $i
}

echo finish`date`