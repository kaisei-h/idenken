#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -V
#$ -e dm.log
#$ -o dm.log
#$ -t 1-30:1

source ~/.bashrc
length=439
num=10000
idx=$SGE_TASK_ID
echo w50 index $idx start `date`

cd /home/kaisei-h/data/rbert
python3 make_seq.py $num $length $idx
cd /home/kaisei-h/raccess
./src/raccess/run_raccess -outfile=/home/kaisei-h/data/rbert/w50/output/out$idx.txt -seqfile=/home/kaisei-h/data/rbert/w50/sequence/seq$idx.fa -access_len=5 -max_span=50
cd /home/kaisei-h/data/rbert
python3 acc_label.py $num $length $idx

echo w50 index $idx finish `date`