#!/bin/bash
#SBATCH -J train_dlmo_acc%a_rayleigh_hvd
#SBATCH -o slurm_out/train_dlmo_acc%a_rayleigh_hvd.o%A
		#SBATCH --partition=medium
#SBATCH --constraint=gpu_mem_80
		# Resources needed for job:
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
env | egrep -i 'GPU|CUDA' | sort
nvidia-smi -L
nvidia-smi
NGPUS=$SLURM_GPUS_ON_NODE
echo "NGPUS:"$NGPUS

echo
echo "==== setup cuda environment"
conda activate dlmo # with horovod build
echo
echo "====" `date +%Y%m%d-%H%M%S` "begin GPU sample programs from the CUDA toolkit"
echo

echo "The model is being trained on ***RAYLEIGH*** task."
echo

# Get start of job information
START_TIME=`date +%s`
host_node=$SLURMD_NODENAME
PY_FILE=dlmo_train_hvd.py
PRETRAIN_PATH=../demo5/DLMO_test/trained_model/mri_dlmo_acc_1_hvd/hvd_cpts/
PRETRAIN_EPOCH=170
ACC=4

PRETRAIN_PATH=../../demo5/DLMO_test/trained_model/mri_cnn_dlmo_acc_1_hvd/
TRAIN_DATA_PATH= #train path with 160k accelerated MR recon images
VAL_DATA_PATH= #tuning path with 8000 accelerated MR recon images
OUTPUT_FLD_PATH=trained_model/mri_cnn_dlmo_acc_
# parameters used in https://arxiv.org/abs/2602.22535
time NCCL_DEBUG=INFO horovodrun -np $NGPUS -H localhost:$NGPUS python ${PY_FILE}\
--acceleration $ACC \
--nepochs 50  \
--train-data-path $TRAIN_DATA_PATH \
--val-data-path $VAL_DATA_PATH \
--output-path $OUTPUT_FLD_PATH \
--batch-size 160 \
--val-batch-size 250 \
--shuffle_patches \
--pretrained-model-path ${PRETRAIN_PATH} \
--pretrained-model-epoch ${PRETRAIN_EPOCH} \
--save-log-ckpts \
--log-file-format log_${NGPUS}_gpus.hdf5

# Get end of job information
END_TIME=`date +%s`
ELAPSED_TIME=$(( $END_TIME - $START_TIME ))
echo "EXIT_STATUS=$EXIT_STATUS"
echo "ELAPSED_TIME=$ELAPSED_TIME"

echo
echo "====" `date +%Y%m%d-%H%M%S` "end GPU samples: EXIT_STATUS=$EXIT_STATUS || ELAPSED_TIME=$ELAPSED_TIME (in sec)"
