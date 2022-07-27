#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -V
#$ -e dm.log
#$ -o dm.log
#$ -t 1-246:1

source ~/.bashrc
idx=$SGE_TASK_ID
echo human index $idx start `date`

# cd /home/kaisei-h/raccess
# ./src/raccess/run_raccess -outfile=/home/kaisei-h/data/GENCODE/human/out$idx.txt -seqfile=/home/kaisei-h/data/GENCODE/human/seq$idx.fa -access_len=5 -max_span=100
cd /home/kaisei-h/data/GENCODE/human
python3 acc_label.py $idx

echo human index $idx finish `date`
