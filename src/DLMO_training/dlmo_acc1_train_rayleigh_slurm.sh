#!/bin/bash
#SBATCH -J train_dlmo_acc1_rayleigh_hvd
#SBATCH -o slurm_out/train_dlmo_acc1_rayleigh_hvd.o%A
		#SBATCH --partition=medium
#SBATCH --constraint=gpu_mem_80
		# Resources needed for job:
#SBATCH --account=cdrhid0024		# Project ID
#SBATCH --gres=gpu:4			# number of gpus
#SBATCH --mem=400G			# memory limit
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=90:00:00			# total run time limit (HH:MM:SS)
		#SBATCH --nodelist=bc002

echo "====" `date +%Y%m%d-%H%M%S` "begin job $SLURM_JOB_NAME ($SLURM_JOB_ID $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID) on node $SLURMD_NODENAME on cluster $SLURM_CLUSTER_NAME"

echo
echo "==== examine the GPU environment"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
#export NVIDIA_VISIBLE_DEVICES=2
env | egrep -i 'GPU|CUDA' | sort
nvidia-smi -L
nvidia-smi
NGPUS=$SLURM_GPUS_ON_NODE
echo "NGPUS:"$NGPUS

echo
echo "==== setup cuda environment"
source /home/prabhat.kc/anaconda3/base_env.sh
source /home/prabhat.kc/anaconda3/horovod_sm80_env.sh
echo
echo "====" `date +%Y%m%d-%H%M%S` "begin GPU sample programs from the CUDA toolkit"
echo

echo "The model is being trained on ***RAYLEIGH*** task."
echo

# Get start of job information
START_TIME=`date +%s`
host_node=$SLURMD_NODENAME
PY_FILE=/projects01/didsr-aiml/zitong.yu/DLMO_demo/DLMO_training/dlmo_train_hvd.py

time horovodrun -np 4 -H localhost:4 python ${PY_FILE} --task rayleigh \
--acceleration 1 \
--batch-size 160 \
--val-batch-size 250 \
--shuffle_patches \
--save-log-ckpts \
--log-file-format log.hdf5


# Get end of job information
END_TIME=`date +%s`
ELAPSED_TIME=$(( $END_TIME - $START_TIME ))
echo "EXIT_STATUS=$EXIT_STATUS"
echo "ELAPSED_TIME=$ELAPSED_TIME"

echo
echo "====" `date +%Y%m%d-%H%M%S` "end GPU samples: EXIT_STATUS=$EXIT_STATUS || ELAPSED_TIME=$ELAPSED_TIME (in sec)"
