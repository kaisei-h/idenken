#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -V
#$ -e dm.log
#$ -o dm.log
#$ -t 1-10000:1

source ~/.bashrc
length=439
num=10000
idx=$SGE_TASK_ID
echo lengthfree index $idx start `date`

cd /home/kaisei-h/data/rbert
python3 make_seq.py $num $length $idx
cd /home/kaisei-h/raccess
./src/raccess/run_raccess -outfile=/home/kaisei-h/data/rbert/output/out$idx.txt -seqfile=/home/kaisei-h/data/rbert/sequence/seq$idx.fa -access_len=5 -max_span=100
cd /home/kaisei-h/data/rbert
python3 acc_label.py $idx

echo lengthfree index $idx finish `date`
