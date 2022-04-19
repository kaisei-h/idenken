#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -V
#$ -e dm.log
#$ -o dm.log
#$ -t 1-500000:10000

source ~/.bashrc
echo start`date`
length=439

cd /home/kaisei-h/data/r-make
for ((i = $SGE_TASK_ID; i < $SGE_TASK_ID+10000; i++)){
	python3 make_seq.py $i $length
	cd /home/kaisei-h/raccess
	./src/raccess/run_raccess -outfile=/home/kaisei-h/data/r-make/r-out/out$i.txt -seqfile=/home/kaisei-h/data/r-make/random/seq$i.fa -access_len=5 -max_span=100
	cd /home/kaisei-h/data/r-make
	python3 acc_label.py $i $length
}

echo finish`date`