#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -V
#$ -e dm.log
#$ -o dm.log
#$ -t 1-50000:1000

source ~/.bashrc
echo start`date`
length=512

cd /home/kaisei-h/data/p-make
for ((i = $SGE_TASK_ID; i < $SGE_TASK_ID+1000; i++)){
	python3 make_seq.py $i $length
	cd /home/kaisei-h/ParasoR/src
	./ParasoR --pre --input /home/kaisei-h/data/p-make/random/seq$i.txt --constraint=100 >> /home/kaisei-h/data/p-make/p-out/out$i.txt
	cd /home/kaisei-h/data/p-make
	python3 bpp_label.py $i
}

echo finish`date`