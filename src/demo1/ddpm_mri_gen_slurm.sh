#!/bin/bash

#SBATCH -J ddpm_mri_slurm
#SBATCH -o ddpm_slurm_out/diff_out_sm.o%A
#SBATCH --partition=medium
		#SBATCH --constraint=gpu_cc_70

# Resources needed for job:
#SBATCH --account=		# Project ID
#SBATCH --gres=gpu:1			# number of gpus
#SBATCH --mem=11G			# memory limit
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=95:00:00			# total run time limit (HH:MM:SS)
		#SBATCH --nodelist=bc005

echo "====" `date +%Y%m%d-%H%M%S` "begin job $SLURM_JOB_NAME ($SLURM_JOB_ID $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID) on node $SLURMD_NODENAME on cluster $SLURM_CLUSTER_NAME"

echo
echo "==== examine the GPU environment"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
#export CUDA_VISIBLE_DEVICES=2
#export NVIDIA_VISIBLE_DEVICES=2
env | egrep -i 'GPU|CUDA' | sort
nvidia-smi -L
nvidia-smi

echo
echo "==== setup cuda environment"
#conda activate ddpm
echo
echo "====" `date +%Y%m%d-%H%M%S` "begin GPU sample programs from the CUDA toolkit"
echo "NGPUS:"$NSLOTS

# Get start of job information
START_TIME=`date +%s`


#conda activate ddpm
#cd src/demo1
# Python file and its input and outputs -----------------------------------------------------------------------------
PY_FILE=scripts/image_sample_newdataset2_centercrop.py
MODEL_PATH=trained_DDPM_model/ema_0.9999_1100000.pt
OUTPUT_FLD="test_out_200k"

# Important ddpm parameters --------------------------
dSTEP=100
nSAMPLES=1
bSIZE=1

# CMD_ARGUMENTS aligning with how we trained the DDPM model -----------------------------------------------------------------------------
MODEL_FLAGS="--image_size 384 --attention_resolutions 32,16,8 --num_channels 128 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_scale_shift_norm True --learn_sigma True"
DIFFUSION_FLAGS="--diffusion_steps ${dSTEP} --noise_schedule cosine "
sample_FLAGS="--save_dir ${OUTPUT_FLD}/HCP_brain_384x384_cropped_260x311_step1100k_ema_samples/ --num_samples ${nSAMPLES} --batch_size ${bSIZE}"

python ${PY_FILE} --model_path ${MODEL_PATH} $MODEL_FLAGS ${DIFFUSION_FLAGS} ${sample_FLAGS} 

# Get end of job information
END_TIME=`date +%s`
ELAPSED_TIME=$(( $END_TIME - $START_TIME ))
echo "EXIT_STATUS=$EXIT_STATUS"
echo "ELAPSED_TIME=$ELAPSED_TIME"

echo
echo "====" `date +%Y%m%d-%H%M%S` "end GPU samples: EXIT_STATUS=$EXIT_STATUS ELAPSED_TIME=$ELAPSED_TIME"
