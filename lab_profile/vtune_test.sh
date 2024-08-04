#!/bin/bash
#SBATCH --job-name vtune-test
#SBATCH --account ACD113087
#SBATCH --output %x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --time 00:10:00
#SBATCH --partition ctest

HPCCAMP_PROFILE=$HOME/lab_profile
#OMP_NUM_THREADS=$SLURM_JOB_CPUS_PER_NODE

source /work/HPC_SCAMP/opt/intel/oneapi/setvars.sh

cd $HPCCAMP_PROFILE

# compile
gcc -o matrix_sum matrix_sum.c 

# clear vtune output directory
rm -fr $HPCCAMP_PROFILE/result/matrix-sum

# run matrix sum with vtune
vtune -collect performance-snapshot \
-r $HPCCAMP_PROFILE/result/matrix-sum \
./matrix_sum
