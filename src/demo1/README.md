# Object generation using DDPM

The code below inputs a trained DDPM model to generates a large batch of image samples that are saved as a numpy array. 

### Usage:

```
conda activate ddpm
cd src/demo1
# Python file and its input and outputs -----------------------------------------------------------------------------
PY_FILE=scripts/image_sample_newdataset2_centercrop.py
MODEL_PATH=trained_DDPM_model/ema_0.9999_1100000.pt
OUTPUT_FLD="test_out_200k"

# Important ddpm parameters --------------------------
dSTEP=1000
nSAMPLES=1
bSIZE=1

# CMD_ARGUMENTS aligning with how we trained the DDPM model -----------------------------------------------------------------------------
MODEL_FLAGS="--image_size 384 --attention_resolutions 32,16,8 --num_channels 128 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_scale_shift_norm True --learn_sigma True"
DIFFUSION_FLAGS="--diffusion_steps ${dSTEP} --noise_schedule cosine "
sample_FLAGS="--save_dir ${OUTPUT_FLD}/HCP_brain_384x384_cropped_260x311_step1100k_ema_samples/ --num_samples ${nSAMPLES} --batch_size ${bSIZE}"

time python ${PY_FILE} --model_path ${MODEL_PATH} $MODEL_FLAGS ${DIFFUSION_FLAGS} ${sample_FLAGS} 
```

### Additional Guides:

1. The DDPM was trained by use of the HCP's young adult dataset[^refDDPM] :

	* This dataset consists of 1,113 subjects  scanned on a customized Siemens 3T MRI system.
	* From each patient, 10 axial slices within their Cerebrospinal fluid (CSF) regions were extracted to train the DDPM model. 

2. No. of diffusion steps is an important parameter when running this ddpm MRI data generation code. 

	![MR images for different diffusion steps](../../docs/pics/diffusion_steps)
<!--[An example of DDPM generated object](../../docs/obj_0.png "An example of DDPM generated object") -->


[^refDDPM]: J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” Advances in neural information processing systems, vol. 33, pp. 6840–6851, 2020.