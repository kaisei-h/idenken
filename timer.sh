#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -V
#$ -e timer5man.log
#$ -o timer5man.log
#$ -t 1-10:1

source ~/.bashrc


cd /home/kaisei-h/project/data/makedata/val
python3 make_seq.py $SGE_TASK_ID 50000
python3 to_index.py $SGE_TASK_ID 50000
cd /home/kaisei-h/raccess
time ./src/raccess/run_raccess -outfile=/home/kaisei-h/project/data/makedata/val/random_out/out_$SGE_TASK_ID.txt -seqfile=/home/kaisei-h/project/data/makedata/val/random/sample_$SGE_TASK_ID.txt -access_len=5 -max_span=20
cd /home/kaisei-h/project/data/makedata/val
python3 acc_label.py $SGE_TASK_ID 50000

