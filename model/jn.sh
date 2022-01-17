#!/bin/bash
#$ -o jnote.log
#$ -e jnote.log
#$ -cwd
#$ -V
#$ -m be
#$ -l gpu
#$ -l cuda=1
#$ -l s_vmem=96G
#$ -l mem_req=96G
#$ -N jnote
#$ -l d_rt=240:00:00
#$ -l s_rt=240:00:00
module load singularity
module load cuda90/toolkit/9.0.176
module load gcc5/5.5.0
singularity run --nv -e ./torch_jn_latest.sif