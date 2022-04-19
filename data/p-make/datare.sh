#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -V
#$ -e re.log
#$ -o re.log

source ~/.bashrc
length=512

array=(47890 5939 715 34931 46975 11711 28927 33934 29962 29969 7965 40897 41899 21931 10836 11896)

cd /home/kaisei-h/data/p-make
for i in ${array[@]}
do
	echo $i
	python3 make_seq.py $i $length
	cd /home/kaisei-h/ParasoR/src
	./ParasoR --pre --input /home/kaisei-h/data/p-make/random/seq$i.txt --constraint=100 >> /home/kaisei-h/data/p-make/p-out/out$i.txt
	cd /home/kaisei-h/data/p-make
	python3 bpp_label.py $i
done
