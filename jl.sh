#!/bin/bash
#$ -o jlab.log
#$ -e jlab.log
#$ -cwd
#$ -V
#$ -m be
#$ -l s_vmem=64G
#$ -l mem_req=64G
#$ -N jlab
#$ -l d_rt=240:00:00
#$ -l s_rt=240:00:00
module load singularity
module load gcc5/5.5.0
singularity run --nv -e ./jlab_latest.sif