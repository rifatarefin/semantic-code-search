#!/bin/bash
#SBATCH -J csnet_gpt2_gtx           # Job name
#SBATCH -o csnet_gpt2_gtx.out       # Name of stdout output file
#SBATCH -e csnet_gpt2_gtx.err       # Name of stderr error file
#SBATCH -p gtx          # Queue (partition) name
#SBATCH -N 1               # Total # of nodes (must be 1 for serial)
#SBATCH -n 1               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 01:30:00        # Run time (hh:mm:ss)
#SBATCH --mail-user=fotios.lygerakis@mavs.uta.edu
#SBATCH --mail-type=all    # Send email at begin and end of job

module load tacc-singularity

module list

cd $WORK/semantic-code-search_gpt2_tf #change this line  based on where you cloned the project
pwd
date

#nvidia-smi
# Launch serial code...
singularity exec script/csnet_gpu.sif python -c 'import tensorflow as tf; print("Tensrflow Version: " + tf.__version__)'

singularity exec script/csnet_gpu.sif python -c 'import torch; print("Torch Version: "  + torch.__version__)'


#python clgen.py  --config /work/05359/sohil777/maverick2/TestDeepFuzz/clgen/tests/data/tiny/config.pbtxt         # Do not use ibrun or any other MPI launcher
#change this as per simulink configuration file. 
#scp -r /tmp/experiments $WORK # THIS IS IMPORTANT DO NOT REMOVE
# ---------------------------------------------------
