#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -V
#$ -e t_error.log
#$ -o t_out.log

source ~/.bashrc

cd /home/kaisei-h/raccess
time for ((i = 0; i < 10000; i++)){
	./src/raccess/run_raccess -outfile=/home/kaisei-h/project/data/makedata/test_long/random_out/out_$i.txt -seqfile=/home/kaisei-h/project/data/makedata/test_long/random/sample_$i.txt -access_len=5 -max_span=20
}

