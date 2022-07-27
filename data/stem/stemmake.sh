#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -V
#$ -e dm.log
#$ -o dm.log
#$ -t 1-1000:1

source ~/.bashrc
length=440
num=10000
idx=$SGE_TASK_ID
echo index $idx start `date`

cd /home/kaisei-h/data/rbert/stem
python3 make_seq.py $num $length $idx
cd /home/kaisei-h/raccess
./src/raccess/run_raccess -outfile=/home/kaisei-h/data/rbert/stem/output/out$idx.txt -seqfile=/home/kaisei-h/data/rbert/stem/sequence/seq$idx.fa -access_len=5 -max_span=100
cd /home/kaisei-h/data/rbert/stem
python3 acc_label.py $idx

echo index $idx finish `date`
