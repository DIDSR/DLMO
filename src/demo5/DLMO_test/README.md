# DLMO test

This example estimates the probability of doublet signal using a trained deep learning-based model observer. It supports the Rayleigh discrimination tasks, and can handle both regular and CNN-denoised images. The script uses Horovod for distributed training and PyTorch for the neural network implementation.

Main components:

  1. Argument parsing
  2. Model loading and initialization
  3. Data loading
  4. Model evaluation
  5. Results saving and AUC calculation

## Step 4: Evaluation-figure generation workflow (`step4_eval.py`)

To apply model observers to the three HDF5 files (rSOS @ 4x, rSOS @ 1x, and U-Net @ 4x) and generate a Figure-4-like plot with PC vs signal length:

```
python step4_eval.py
```

This generates:
- `./dlmo_predictions/figure4_step4.png` — Figure 4 plot
- `./dlmo_predictions/figure4_step4.csv` — PC values by signal length (4-8 mm)

## Single observer run (`dlmo_test_hvd.py`)

Usage:

```
python dlmo_test_hvd.py [-h] [--task TASK] [--test-path TEST_PATH] [--hdf5-file HDF5_FILE] [--is-cnn-denoised] [--test-cnn-denoiser TEST_CNN_DENOISER]
                        [--acceleration ACCELERATION] [--num-channels NUM_CHANNELS] [--batch-size BATCH_SIZE]
                        [--batches-per-allreduce BATCHES_PER_ALLREDUCE] [--fp16-allreduce]
                        [--pretrained-model-path PRETRAINED_MODEL_PATH]
                        [--pretrained-model-checkpoint-format PRETRAINED_MODEL_CHECKPOINT_FORMAT]
                        [--pretrained-model-epoch PRETRAINED_MODEL_EPOCH]
                        [--preds-tag PREDS_TAG]

Arguments:
-h, --help - Show this help message and exit
--task - Task type for testing. Options: detection, rayleigh. Default is 'rayleigh'
--test-path - Path to directory containing images for testing
--hdf5-file - Optional direct path to HDF5 file (overrides --test-path naming)
--is-cnn-denoised - Flag indicating whether the input images have been preprocessed with CNN denoising (flag)
--test-cnn-denoiser - Name of the CNN denoiser model that was used to denoise the input images
--acceleration - Acceleration factor for image reconstruction in range of 2 to 12. Default is 2
--num-channels - Number of image channels for processing. Use 3 for RGB images and 1 for grayscale images. Default is 1
--batch-size - Batch size for testing phase to control memory usage and processing speed
--batches-per-allreduce - Number of batches processed locally before executing allreduce across workers. This multiplies the total batch size (1 loss function equals 1 batches-per-allreduce). Default is 1
--fp16-allreduce - Enable fp16 compression during allreduce operations to reduce communication overhead (flag)
--pretrained-model-path - Path to directory containing the previously trained model for testing
--pretrained-model-checkpoint-format - Format string for checkpoint filenames. Default is 'checkpoint-{epoch}.pth.tar'
--pretrained-model-epoch - Epoch number of the pretrained model to use for transfer learning. Default is 150
--preds-tag - Optional custom suffix for output filename (saved as preds_<tag>.npy)
```

Example:

```
ACC=4

TEST_PATH=../image_acquisition_and_reconstruction/examples/img_w_signal/
TRAINED_MODEL_PATH=./trained_model/mri_cnn_io_acc_${ACC}_hvd/hvd_cpts/
#TRAINED_MODEL_PATH=./trained_model/mri_cnn_io_acc_${ACC}_unet_hvd/hvd_cpts/

# Transfer-learned models
EPOCH=50

# Base model
#EPOCH=170

python dlmo_test_hvd.py --task rayleigh \
--test-path $TEST_PATH \
--acceleration ${ACC} \
--batch-size 6 \
--batches-per-allreduce 1 \
--fp16-allreduce \
--pretrained-model-path $TRAINED_MODEL_PATH \
--pretrained-model-epoch $EPOCH
```

For CNN-denoised testing, set `--is-cnn-denoised` and point `--test-path` to `../AI_rec/ai_rec_prediction`.