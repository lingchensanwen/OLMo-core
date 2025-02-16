#!/bin/bash
#SBATCH -J deepseek1.3b           # Job name
#SBATCH -o slurm-outputs/deepseek1.3b.o%j       # Name of stdout output file
#SBATCH -e slurm-outputs/deepseek1.3b.e%j       # Name of stderr output file
#SBATCH -p gh-dev          # Queue (partition) name
#SBATCH -N 1             # Total # of nodes
#SBATCH -t 1:59:59        # Run time (hh:mm:ss)
#SBATCH -A AST24021       # Allocation name (req'd if you have more than 1)

srun /work/07144/yw23374/vista/OLMo-core/slurm_scripts/run_cpt_multi_node.sh $SLURM_JOB_NODELIST $SLURM_JOB_NUM_NODES