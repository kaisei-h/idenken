#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -V
#$ -e dm.log
#$ -o dm.log
#$ -t 1-04222:1

source ~/.bashrc
idx=$(printf "%05d" $SGE_TASK_ID)
echo RF$idx start `date`

cd /home/kaisei-h/raccess
./src/raccess/run_raccess -outfile=/home/kaisei-h/data/RF/out$idx.txt -seqfile=/home/kaisei-h/data/RF/RF$idx.fa -access_len=5 -max_span=100
cd /home/kaisei-h/data/RF
python3 acc_label.py $idx
rm -f out$idx.txt

echo RF$idx finish `date`
