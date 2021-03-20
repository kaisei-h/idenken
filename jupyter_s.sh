#!/bin/bash
#$ -o jn_out.log
#$ -e jn_error.log
#$ -cwd
#$ -m be
#$ -l gpu
#$ -l cuda=1
#$ -N j_short
module load singularity
module load cuda90/toolkit/9.0.176
module load gcc5/5.5.0
singularity run --nv -e ./torch_jn_latest.sif
