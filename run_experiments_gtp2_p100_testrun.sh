#!/bin/bash
#SBATCH -J csnet_gpt2_p100_testrun           # Job name
#SBATCH -o csnet_gpt2_p100_testrun.out       # Name of stdout output file
#SBATCH -e csnet_gpt2_p100_testrun.err       # Name of stderr error file
#SBATCH -p p100          # Queue (partition) name
#SBATCH -N 1               # Total # of nodes (must be 1 for serial)
#SBATCH -n 1               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 24:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-user=fotios.lygerakis@mavs.uta.edu
#SBATCH --mail-type=all    # Send email at begin and end of job

module load tacc-singularity
module load cuda/10.1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/apps/cuda10_1/cudnn/7.6/lib64

module list

cd $WORK/semantic-code-search_gpt2_tf/script #change this line  based on where you cloned the project
pwd
date

singularity exec --nv --home $(pwd):/home/dev csnet_gpu.sif wandb login a38731f94eccfecd49c3ef107cf5e50291c81fd3

#nvidia-smi
# Launch serial code...
#singularity exec script/csnet_gpu.sif python -c 'import tensorflow as tf; print("Tensrflow Version: " + tf.__version__)'
#singularity exec script/csnet_gpu.sif python -c 'import torch; print("Torch Version: "  + torch.__version__)'

singularity exec --nv --home $(pwd):/home/dev csnet_gpu.sif python ../src/train.py --model gpt2 --testrun ../resources/saved_models/ ../resources/data/python/final/jsonl/train/ ../resources/data/python/final/jsonl/valid/ ../resources/data/python/final/jsonl/test

#singularity exec --nv --home $(pwd):/home/dev csnet_gpu.sif python -c "import torch; print(torch.__version__)"
#singularity exec --nv --home $(pwd):/home/dev csnet_gpu.sif python -c "import torch; print(torch.cuda.is_available())"

#python clgen.py  --config /work/05359/sohil777/maverick2/TestDeepFuzz/clgen/tests/data/tiny/config.pbtxt         # Do not use ibrun or any other MPI launcher
#change this as per simulink configuration file. 
#scp -r /tmp/experiments $WORK # THIS IS IMPORTANT DO NOT REMOVE
# ---------------------------------------------------
