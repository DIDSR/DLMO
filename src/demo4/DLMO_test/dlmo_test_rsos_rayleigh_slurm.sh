#!/bin/bash
#SBATCH -J dlmo_test_rayleigh_acc%a_hvd
#SBATCH -o slurm_out/dlmo_test_rayleigh_acc%a_hvd.o%A
		# Resources needed for job:
#SBATCH --account=cdrhid0024		# Project ID
#SBATCH --gres=gpu:a100:1			# number of gpus
#SBATCH --mem=400G			# memory limit
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=6:00:00			# total run time limit (HH:MM:SS)

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
source /anaconda3/base_env.sh
source /anaconda3/horovod_sm80_env.sh
echo
echo "====" `date +%Y%m%d-%H%M%S` "begin GPU sample programs from the CUDA toolkit"
echo

#echo "The model is being trained on ***RAYLEIGH*** task."
#echo "The model is being trained on ***DETECTION*** task."
#echo

# Get start of job information
START_TIME=`date +%s`
host_node=$SLURMD_NODENAME
ACC=$SLURM_ARRAY_TASK_ID


TEST_PATH=../synthetic_data_generation/examples/img_w_signal/
TRAINED_MODEL_PATH=./trained_model/mri_cnn_io_acc_${ACC}_hvd/hvd_cpts/
#TRAINED_MODEL_PATH=./trained_model/mri_cnn_io_acc_${ACC}_unet_hvd/hvd_cpts/

# Transfer-learned models
EPOCH=50

# Based model
#EPOCH=170

PY_FILE=dlmo_test_hvd.py

horovodrun -np 1 -H localhost:1 python ${PY_FILE} --task rayleigh \
--test-path $TEST_PATH \
--acceleration ${ACC} \
--batch-size 6 \
--batches-per-allreduce 1 \
--fp16-allreduce \
--pretrained-model-path $TRAINED_MODEL_PATH \
--pretrained-model-epoch $EPOCH


# Get end of job information
END_TIME=`date +%s`
ELAPSED_TIME=$(( $END_TIME - $START_TIME ))
echo "EXIT_STATUS=$EXIT_STATUS"
echo "ELAPSED_TIME=$ELAPSED_TIME"

echo
echo "====" `date +%Y%m%d-%H%M%S` "end GPU samples: EXIT_STATUS=$EXIT_STATUS || ELAPSED_TIME=$ELAPSED_TIME (in sec)"
