#!/bin/bash
#SBATCH -J csnet_gpt2_distr           # Job name
#SBATCH -o log/csnet_gpt2_v100.out       # Name of stdout output file
#SBATCH -e log/csnet_gpt2_v100.err       # Name of stderr error file
#SBATCH -p p100          # Queue (partition) name
#SBATCH -N 1               # Total # of nodes (must be 1 for serial)
#SBATCH -n 1               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 24:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-user=rifatarefin19@gmail.com
#SBATCH --mail-type=all    # Send email at begin and end of job

# module load tacc-singularity
# module load cuda/10.1
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/apps/cuda10_1/cudnn/7.6/lib64

module list
cd $WORK2/semantic-code-search/src #change this line  based on where you cloned the project
pwd
date
# singularity exec --nv --home $(pwd):/home/dev csnet_gpu_mpi.sif wandb login 24aacfdb048a7a2e59a54fb3258ec0b67b8eb013

#nvidia-smi
# Launch serial code...
#singularity exec script/csnet_gpu.sif python -c 'import tensorflow as tf; print("Tensrflow Version: " + tf.__version__)'
#singularity exec script/csnet_gpu.sif python -c 'import torch; print("Torch Version: "  + torch.__version__)'
# export SINGULARITY_BINDPATH="/opt/"

ibrun -np 1 python3 train.py --model selfatt
# --evaluate-model /work2/07782/marefin/maverick2/semantic-code-search/resources/saved_models/gpt2-2021-06-12-15-47-47_model_best.pkl.gz
#--run-name gpt2-2021-06-21-12-02-57
#--run-name GPT-2-2021-06-06-11-28-33

#../resources/saved_models/ ../resources/data/python/final/jsonl/train/ ../resources/data/python/final/jsonl/valid/ ../resources/data/python/final/jsonl/test

# singularity exec --nv --home $(pwd):/home/dev csnet_gpu.sif python -c "import torch; print(torch.__version__)"
# singularity exec --nv --home $(pwd):/home/dev csnet_gpu.sif python -c "import torch; print(torch.cuda.is_available())"

#python clgen.py  --config /work/05359/sohil777/maverick2/TestDeepFuzz/clgen/tests/data/tiny/config.pbtxt         # Do not use ibrun or any other MPI launcher
#change this as per simulink configuration file. 
#scp -r /tmp/experiments $WORK # THIS IS IMPORTANT DO NOT REMOVE
# ---------------------------------------------------
