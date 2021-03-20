#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -V
#$ -e long_data.log
#$ -o long_data.log

source ~/.bashrc


for ((i = 0; i < 5477; i++)){
	cd /home/kaisei-h/raccess
	./src/raccess/run_raccess -outfile=/home/kaisei-h/project/data/RF01210/out_$i.txt -seqfile=/home/kaisei-h/project/data/RF01210/sample_$i.txt -access_len=5 -max_span=20
	cd /home/kaisei-h/project/data/RF01210
	python3 acc_label.py $i
	python3 to_index.py $i
}

