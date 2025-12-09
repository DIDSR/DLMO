#!/bin/bash
#SBATCH -J test_img_generation
#SBATCH -o slurm_out/test_img_generation.o%A
		# Resources needed for job:
#SBATCH --account=cdrhid0024		# Project ID
#SBATCH --gres=gpu:a100:1			# number of gpus
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
source /home/prabhat.kc/anaconda3/ddpm_env.sh
source /home/prabhat.kc/anaconda3/horovod_sm80_env.sh
echo
echo "====" `date +%Y%m%d-%H%M%S` "begin GPU sample programs from the CUDA toolkit"
echo

echo

# Get start of job information
START_TIME=`date +%s`
host_node=$SLURMD_NODENAME
PY_FILE=rsos_ddpm_test.py

time python ${PY_FILE} 1
time python ${PY_FILE} 4
time python ${PY_FILE} 8


# Get end of job information
END_TIME=`date +%s`
ELAPSED_TIME=$(( $END_TIME - $START_TIME ))
echo "EXIT_STATUS=$EXIT_STATUS"
echo "ELAPSED_TIME=$ELAPSED_TIME"

echo
echo "====" `date +%Y%m%d-%H%M%S` "end GPU samples: EXIT_STATUS=$EXIT_STATUS || ELAPSED_TIME=$ELAPSED_TIME (in sec)"
