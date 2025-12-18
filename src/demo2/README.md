# Object generation using DDPM

This example generates a large batch of image samples from a model and save them as a large numpy array. This can be used to produce samples for FID evaluation.

Usage:

```
OUTPUT_FLD="./test_out_100k/"
PY_FILE=./script/image_sample_newdataset2_centercrop.py
MODEL_PATH=./trained_DDPM_model/ema_0.9999_1100000.pt

#CMD_ARGUMENTS
MODEL_FLAGS="--image_size 384 --attention_resolutions 32,16,8 --num_channels 128 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_scale_shift_norm True --learn_sigma True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule cosine "
sample_FLAGS="--save_dir ${OUTPUT_FLD}/HCP_brain_384x384_cropped_260x311_step1100k_ema_samples/ --num_samples 10000 --batch_size 8"

python ./script/image_sample_newdataset2_centercrop.py --model_path ${MODEL_PATH} $MODEL_FLAGS $DIFFUSION_FLAGS $sample_FLAGS

```

Output files are in `${OUTPUT_FLD}` folder as `.npz` files.
