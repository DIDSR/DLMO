#!/bin/bash
#SBATCH -J TrUnet_acc%a_rayleigh_hvd
#SBATCH -o slurm_out/TrUnet_acc%a_rayleigh_hvd.o%A
		#SBATCH --partition=medium
		# Resources needed for job:
#SBATCH --account=cdrhid0024		# Project ID
#SBATCH --gres=gpu:a100:2			# number of gpus
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

echo "The model is being trained with data on ***RAYLEIGH*** task."
echo

# Get start of job information
START_TIME=`date +%s`
host_node=$SLURMD_NODENAME
PY_FILE=/projects01/didsr-aiml/zitong.yu/DLMO/src/ai_rec/unet/DL_denoiser_train.py
ACC=$SLURM_ARRAY_TASK_ID

time horovodrun -np 2 -H localhost:2 python $PY_FILE --task rayleigh \
--training-path /projects01/didsr-aiml/zitong.yu/DLMO/raw_data/ \
--val-path /projects01/didsr-aiml/zitong.yu/DLMO/raw_data/ \
--acceleration $ACC \
--model_name unet \
--num-channels 1 \
--batch-size 32 \
--base-lr 1e-4 \
--max-epochs 150 \
--val-batch-size 32 \
--batches-per-allreduce 1 \
--shuffle_patches \
--fp16-allreduce \
--save-log-ckpts \
--log-file-format log.hdf5


# Get end of job information
END_TIME=`date +%s`
ELAPSED_TIME=$(( $END_TIME - $START_TIME ))
echo "EXIT_STATUS=$EXIT_STATUS"
echo "ELAPSED_TIME=$ELAPSED_TIME"

echo
echo "====" `date +%Y%m%d-%H%M%S` "end GPU samples: EXIT_STATUS=$EXIT_STATUS || ELAPSED_TIME=$ELAPSED_TIME (in sec)"
