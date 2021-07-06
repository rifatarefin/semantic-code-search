#!/bin/bash
#SBATCH -J csnet           # Job name
#SBATCH -o log/test.out       # Name of stdout output file
#SBATCH -e log/test.err       # Name of stderr error file
#SBATCH -p gtx          # Queue (partition) name
#SBATCH -N 2               # Total # of nodes (must be 1 for serial)
#SBATCH -n 2               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 0:30:00        # Run time (hh:mm:ss)
#SBATCH --mail-user=ratt.m1913@gmail.com
#SBATCH --mail-type=all    # Send email at begin and end of job

singularity exec HOROVOD_CUDA_HOME=$TACC_CUDA_DIR HOROVOD_NCCL_HOME=$TACC_NCCL_DIR CC=gcc \
    HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_GPU_BROADCAST=NCCL HOROVOD_WITH_TENSORFLOW=1 pip3 install \
    --user horovod==0.19.2 --no-cache-dir

cd $WORK/semantic-code-search/script

ibrun singularity run --nv --home $(pwd):/home/dev csnet_gpu_mpi.sif python ../src/train.py --model gpt2 --dryrun --testrun ../resources/saved_models/ ../resources/data/python/final/jsonl/train/ ../resources/data/python/final/jsonl/valid/ ../resources/data/python/final/jsonl/test