#!/bin/bash
HOST=$1
NODES=$2

# Load modules
module purge  # Ensure a clean environment
module load gcc/13.2.0 cuda/12.4 nccl/12.4 nvidia_math/12.4
module load python3
module load tacc-apptainer
echo "Loaded Modules:"
module list  # Now, this will correctly list loaded modules

export CC=gcc

export CXX=g++

export TORCH_CUDA_ARCH_LIST="9.0 9.0a"

cd $SCRATCH || exit
apptainer  pull docker://nvcr.io/nvidia/pytorch:24.12-py3
apptainer shell --nv \
pytorch_24.12-py3.sif


# Move to project directory
cd ${WORK}/OLMo-core || exit

# pip install -e .[all] #need to install directly, conda may have conflict

# Set environment variables
export WANDB_MODE=online
export NCCL_DEBUG=INFO
export NODENAME=$(hostname -s)

export RANK=$SLURM_PROCID
export FS_LOCAL_RANK=$SLURM_PROCID
export LOCAL_WORLD_SIZE=1  # $SLURM_NTASKS_PER_NODE
export LOCAL_RANK=0  # $SLURM_LOCALID
export NODE_RANK=$((($RANK - $LOCAL_RANK) / $LOCAL_WORLD_SIZE))

# Debugging info
echo "----------------------------------------------------"
echo " SLURM Job Info "
echo "----------------------------------------------------"
echo "Nodelist       : $SLURM_JOB_NODELIST"
echo "Nodes          : $SLURM_JOB_NUM_NODES"
echo "Tasks Per Node : $SLURM_NTASKS_PER_NODE"
echo "Node Rank      : $NODE_RANK"
echo "World Size     : $WORLD_SIZE"
echo "----------------------------------------------------"

# Master port and address
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$SLURM_NNODES
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

echo "MASTER_PORT=$MASTER_PORT"
echo "WORLD_SIZE=$WORLD_SIZE"
echo "MASTER_ADDR=$MASTER_ADDR"

# NCCL optimizations
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCCL_MIN_CHANNELS=32
export NCCL_DEBUG=INFO  # Debugging NCCL

# Debugging
echo "Starting training..."
sleep 5  # Just to ensure the env is fully loaded

# Run training
torchrun --nproc_per_node=1 \
    --rdzv-backend=c10d \
    --node_rank=${NODE_RANK} \
    --rdzv_conf 'read_timeout=420' \
    --nnodes="${WORLD_SIZE}" \
    --rdzv_id 12349 \
    --rdzv_endpoint "${MASTER_ADDR}:${MASTER_PORT}" \
    --master_addr ${MASTER_ADDR} \
    src/examples/deepseek/train.py run_name 
