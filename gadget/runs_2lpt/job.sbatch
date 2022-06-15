#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=40
#SBATCH --job-name=gadget-run11
#SBATCH --partition=caps
#SBATCH --output=%x.output
#SBATCH --error=%x.errors
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=aaronjo2@illinois.edu

echo
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."
echo "Current working directory is `pwd`"
echo


source ../.gadget_env.sh

srun ./Gadget4 param_run11.txt
#mpiexec -np $SLURM_NPROCS ./Gadget4 param.txt
