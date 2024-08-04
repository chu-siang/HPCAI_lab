#!/bin/bash
#SBATCH -A ACD113087        # Account name/project number
#SBATCH -J hpl          # Job name
#SBATCH -p ctest         # Partition name
#SBATCH -n 25              # Number of MPI tasks (i.e. processes)
#SBATCH -c 1            # Number of cores per MPI task
#SBATCH -t 10:00        # Wall time limit (days-hrs:min:sec)

export MODULEPATH=$MODULEPATH:/work/HPC_SCAMP/opt/modulefiles/twnia2
module purge
module load openmpi/4.1.5

source /work/HPC_SCAMP/opt/intel/oneapi/setvars.sh
# {TODO1} go to hpl directory
cd $HOME/Day3_benchmark/cpu/hpl
mpirun -np 25 --bind-to none ./xhpl >> hpl_result.txt

HPCCAMP_PROFILE=$HOME/Day3_benchmark/cpu/hpl
# {TODO2} use vtune command to profile hpl
vtune  -collect hotspots -r $HPCCAMP_PROFILE/vtune_result40  -- mpirun -np 25 --bind-to none ./xhpl