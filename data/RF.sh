#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -V

source ~/.bashrc
echo start `date`

cd /home/kaisei-h/raccess
./src/raccess/run_raccess -outfile=/home/kaisei-h/data/rbert/output/out100001.txt -seqfile=/home/kaisei-h/data/RF00001.fa -access_len=5 -max_span=100
./src/raccess/run_raccess -outfile=/home/kaisei-h/data/rbert/output/out100002.txt -seqfile=/home/kaisei-h/data/RF00002.fa -access_len=5 -max_span=100
cd /home/kaisei-h/data/rbert
python3 acc_label.py 100001
python3 acc_label.py 100002

echo w100 index $idx finish `date`
