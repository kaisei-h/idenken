#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -V
#$ -l d_rt=240:00:00
#$ -l s_rt=240:00:00

source ~/.bashrc
echo start `date`
cd /home/kaisei-h/raccess
time ./src/raccess/run_raccess -outfile=/home/kaisei-h/project/data/real_data/mouse_retry.txt -seqfile=/home/kaisei-h/project/data/real_data/gencode.vM28.transcripts.fa -access_len=5 -max_span=100
echo finish`date`