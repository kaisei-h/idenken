#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -V
#$ -e long_data.log
#$ -o long_data.log
#$ -t 1-500000:1000

source ~/.bashrc

for ((i = $SGE_TASK_ID; i < $SGE_TASK_ID+1000; i++)){
	cd /home/kaisei-h/project/data/makedata/train
	python3 make_seq.py $i
	python3 to_index.py $i
	cd /home/kaisei-h/raccess
	./src/raccess/run_raccess -outfile=/home/kaisei-h/project/data/makedata/train/random_out/out_$i.txt -seqfile=/home/kaisei-h/project/data/makedata/train/random/sample_$i.txt -access_len=5 -max_span=100
	cd /home/kaisei-h/project/data/makedata/train
	python3 acc_label.py $i
}

