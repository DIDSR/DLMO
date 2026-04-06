# DLMO test

This example estimates the probability of doublet signal using a trained deep learning-based model observer. It supports the Rayleigh discrimination tasks, and can handle both regular and CNN-denoised images.

1. Concretely, it takes pretrained DLMO weights and MR-reconstructed images as input, and outputs the probability of whether the MR image contains doublet or singlet signals.
2. Re-run this script by updating paths for different reconstruction methods to obtain the corresponding DLMO discrimination outputs.

## Single observer run (`dlmo_test_hvd.py`)

Usage:

```
python dlmo_test_hvd.py [-h] [--task TASK] [--test-path TEST_PATH]
                        [--cnn-denoiser-name CNN_DENOISER_NAME]
                        [--acceleration ACCELERATION]
                        [--num-channels NUM_CHANNELS]
                        [--batch-size BATCH_SIZE]
                        [--pretrained-model-path PRETRAINED_MODEL_PATH]
                        [--pretrained-model-checkpoint-format PRETRAINED_MODEL_CHECKPOINT_FORMAT]
                        [--pretrained-model-epoch PRETRAINED_MODEL_EPOCH]
                        [--out-tag OUT_TAG]

Applying a trained DLMO for detection/descrimination task

Arguments:
  -h, --help                                      show this help message and exit
  --task TASK                                     Task type (detection/rayleigh).
  --test-path TEST_PATH                           Path to reconstructed MR images with signals.
  --cnn-denoiser-name CNN_DENOISER_NAME           Name of cnn denoiser that denoised the MR images.
  --acceleration ACCELERATION                     acceleration factor in range of 2 to 12.
  --num-channels NUM_CHANNELS                     3 for rgb images and 1 for gray scale images
  --batch-size BATCH_SIZE                         Batch size.
  --pretrained-model-path PRETRAINED_MODEL_PATH   The previous trained path to DLMO weights.
  --pretrained-model-checkpoint-format PRETRAINED_MODEL_CHECKPOINT_FORMAT
                                                  checkpoint file format
  --pretrained-model-epoch PRETRAINED_MODEL_EPOCH Epoch number of the pretrained DLMO checkpoint file.
  --out-tag OUT_TAG                               Optional tag for the output filename (saved as
                                                  preds_<out_tag>.npy).
```

Example:

```
ACC=4

TEST_PATH=../../demo3/rsos_rec
TRAINED_MODEL_PATH=./trained_model/mri_cnn_dlmo_acc_${ACC}_hvd/ # for rsos recon
# TRAINED_MODEL_PATH=./trained_model/mri_cnn_dlmo_acc_${ACC}_unet_hvd/ #for unet recon

# Base model for fully sampled DLMO learning from sratch
# EPOCH=170

# Transfer learning is employed for DLMO training for accelerated rSOS and unet recon
EPOCH=50

python dlmo_test_hvd.py --task rayleigh \
--test-path $TEST_PATH \
--acceleration ${ACC} \
--batch-size 10 \
--batches-per-allreduce 1 \
--fp16-allreduce \
--pretrained-model-path $TRAINED_MODEL_PATH \
--pretrained-model-epoch $EPOCH
```