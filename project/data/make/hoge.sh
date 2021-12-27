#$ -cwd
#!/bin/bash
#$ -m be


source ~/.bashrc
echo start `date`
python3 tsv_to_pkl.py
echo finish`date`