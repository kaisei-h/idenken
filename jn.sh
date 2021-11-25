#!/bin/bash
#$ -o j_out.log
#$ -e j_error.log
#$ -cwd
#$ -m be
#$ -l gpu
#$ -l cuda=1
#$ -l s_vmem=48G 
#$ -l mem_req=48G
#$ -N j_serv
#$ -l d_rt=240:00:00
#$ -l s_rt=240:00:00
module load singularity
module load cuda90/toolkit/9.0.176
module load gcc5/5.5.0
singularity run --nv -e ./torch_jn_latest.sif