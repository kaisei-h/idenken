#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -V
#$ -e data.log
#$ -o data.log

source ~/.bashrc

for ((i = 450000; i < 500000; i++)){
	cd /home/kaisei-h/project/data/makedata/test
	python3 acc_label.py $i
}
