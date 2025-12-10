# Evaluating the resolution of AI-based accelerated MR reconstruction using a deep learning-based model observer (DLMO)
**The DLMO framework** evaluates multi-coil sensitivity encoding parallel MR imaging systems at different acceleration factors on the Rayleigh discrimination task as a surrogate measure of resolution, as detailed in our paper. You can implement the DLMO framework with other vendor-specific image acquisition settings and reconstruction methods to demonstrate their diagnostic efficacy.

![Evaluation using the DLMO framework](docs/github_fig1.png "Evaluation using the DLMO framework")

## Requirements

Create a new conda enviroment and install the required packages as follows:
```
$ conda create --name dlmo --file requirements.txt
$ conda activate dlmo
```

## Usage
The example codes below show how to run the DLMO framework. To run the code, please accordingly change input paths and relevant parameters for your application.

* **A simple example of the DLMO framework**

  This example includes four parts: 1) Synthetic data generation, 2) AI reconstruction, 3) DLMO training, and 4) DLMO testing. Please follow the order to run this example.

  1. *Synthetic data generation*

This script performs forward projection and reconstruction of DDPM (Denoising Diffusion Probabilistic Models) generated objects using RSOS (Root Sum of Squares) method to create a few examples of accelerated MR images. It saves the reconstructions in HDF5 format as well as png format.

Command-line Options:
```
Acceleration (int): Acceleration factor for sparse sampling (2, 4, 6, or 8).
```

Usage:
```
python synthetic_img_generation.py [acceleration factor]
```

Example of running the script at acceleration factor 2:
```
python synthetic_img_generation.py 2
```

Note: Ensure that all required data files and directories are properly set up before running the script.

Example of the outputs:

![Object with a doulet signal](src/demo1/synthetic_data_generation/examples/img_w_signal/gt_sample_doublet_rsos_0.png "Object with a doulet signal")
![Fully-sampled MR images with a doulet signal reconstructed using rSOS](src/demo1/synthetic_data_generation/examples/img_w_signal/fully_sampled_doublet_rsos_0.png "Fully-sampled MR images with a doulet signal reconstructed using rSOS")
![Accelerated MR images with a doulet signal reconstructed using rSOS](src/demo1/synthetic_data_generation/examples/img_w_signal/accelerated_sample_doublet_rsos_0.png "Accelerated MR images with a doulet signal reconstructed using rSOS")

  2. *AI reconstruction*

Demo scripts for AI-based reconstruction methods. A U-Net example is included. The test data and its predictions are large files so does not provided here. To obtain those, please generate them using scripts in synthetic_data_generation folder.

Usage:
```
python synthetic_img_generation.py [-h] [--task TASK] [--test-path TEST_PATH] [--acceleration ACCELERATION]
                                   [--model_name MODEL_NAME] [--num-channels NUM_CHANNELS] [--batch-size BATCH_SIZE]
                                   [--batches-per-allreduce BATCHES_PER_ALLREDUCE] [--fp16-allreduce]
                                   [--pretrained-model-path PRETRAINED_MODEL_PATH]
                                   [--pretrained-model-checkpoint-format PRETRAINED_MODEL_CHECKPOINT_FORMAT]
                                   [--pretrained-model-epoch PRETRAINED_MODEL_EPOCH]

Arguments:
--task: Task type (detection/rayleigh). Default is 'rayleigh'.
--test-path: Path to noisy images for testing.
--acceleration: Acceleration factor (2, 4, 6, or 8).
--model_name: CNN denoiser model (cnn3, redcnn, udncnn, dncnn, unet).
--num-channels: Number of channels (1 for grayscale, 3 for RGB). Default is 1.
--batch-size: Batch size for testing.
--batches-per-allreduce: Number of batches processed locally before allreduce. Default is 1.
--fp16-allreduce: Use fp16 compression during allreduce (flag).
--pretrained-model-path: Path to the directory containing the pre-trained model.
--pretrained-model-checkpoint-format: Format of the checkpoint file. Default is 'checkpoint-{epoch}.pth.tar'.
--pretrained-model-epoch: Epoch number of the pre-trained model to use. Default is 150.
```

  3. *DLMO training*

Train the deep learning-based model observer. It supports distributed training using Horovod and handles various configurations through command-line arguments.

Main components:

  1. Argument parsing and setup
  2. Data loading (training and validation)
  3. Model initialization and optimization setup
  4. Training loop with validation
  5. Checkpoint saving and logging

Usage:
```
python dlmo_train_hvd.py [-h] [--task TASK] [--acceleration ACCELERATION] [--pretrained-model-path PRETRAINED_MODEL_PATH]
                         [--pretrained-model-checkpoint-format PRETRAINED_MODEL_CHECKPOINT_FORMAT]
                         [--pretrained-model-epoch PRETRAINED_MODEL_EPOCH] [--batch-size BATCH_SIZE]
                         [--val-batch-size VAL_BATCH_SIZE] [--batches-per-allreduce BATCHES_PER_ALLREDUCE]
                         [--shuffle_patches] [--wd WD] [--fp16-allreduce] [--checkpoint-format CHECKPOINT_FORMAT]
                         [--save-log-ckpts] [--log-file-format LOG_FILE_FORMAT]
Arguments:
-h, --help - Show this help message and exit
--task - Task type for training. Options: detection, rayleigh
--acceleration - Acceleration factor for image reconstruction. Valid options: 1, 2, 4, 6, 8, 10, 12
--pretrained-model-path - Path to directory containing a previously trained model for transfer learning
--pretrained-model-checkpoint-format - Format string for pretrained model checkpoint filenames. Default is 'checkpoint-{epoch}.pth.tar'
--pretrained-model-epoch - Epoch number of the pretrained model to use for transfer learning. Default is 150
--batch-size - Batch size for training phase to control memory usage and processing speed
--val-batch-size - Batch size for validation data processing. Default is 16
--batches-per-allreduce - Number of batches processed locally before executing allreduce across workers. This multiplies the total batch size (1 loss function equals 1 batches-per-allreduce). Default is 1
--shuffle_patches - Enable shuffling of train/validation patch pairs (input-target) at utils.data.DataLoader level rather than at HDF5dataloader level (flag)
--wd - Weight decay value for regularization on weights. Default is 0.0
--fp16-allreduce - Enable fp16 compression during allreduce operations to reduce communication overhead (flag)
--checkpoint-format - Format string for checkpoint filenames during training. Default is 'checkpoint-{epoch}.pth.tar'
--save-log-ckpts - Enable saving of log writer and checkpoint files during training (flag)
--log-file-format - Format string for log filenames. Default is 'log-{epoch}.pkl'
```

Examples:

To training the base model with fully-sampled data (at acceleration factor of 1):

```
python dlmo_train_hvd.py --task rayleigh \
--acceleration 1 \
--batch-size 160 \
--val-batch-size 250 \
--shuffle_patches \
--save-log-ckpts \
--log-file-format log.hdf5
```

To train models with transfer learning at accelerated data:

```
PRETRAIN_PATH=../DLMO_test/trained_model/mri_dlmo_acc_1_hvd/hvd_cpts/
PRETRAIN_EPOCH=170
ACC=4

python dlmo_train_hvd.py --task rayleigh \
--acceleration $ACC \
--batch-size 160 \
--val-batch-size 250 \
--shuffle_patches \
--save-log-ckpts \
--pretrained-model-path ${PRETRAIN_PATH} \
--pretrained-model-epoch ${PRETRAIN_EPOCH} \
--log-file-format log.hdf5
```

  4. *DLMO testing*

This example estimates the probability of doublet signal using a trained deep learning-based model observer. It supports the Rayleigh discrimination tasks, and can handle both regular and CNN-denoised images. The script uses Horovod for distributed training and PyTorch for the neural network implementation.

Main components:

  1. Argument parsing
  2. Model loading and initialization
  3. Data loading
  4. Model evaluation
  5. Results saving and AUC calculation

Usage:

```
python dlmo_test_hvd.py [-h] [--task TASK] [--test-path TEST_PATH] [--is-cnn-denoised] [--test-cnn-denoiser TEST_CNN_DENOISER]
                        [--acceleration ACCELERATION] [--num-channels NUM_CHANNELS] [--batch-size BATCH_SIZE]
                        [--batches-per-allreduce BATCHES_PER_ALLREDUCE] [--fp16-allreduce]
                        [--pretrained-model-path PRETRAINED_MODEL_PATH]
                        [--pretrained-model-checkpoint-format PRETRAINED_MODEL_CHECKPOINT_FORMAT]
                        [--pretrained-model-epoch PRETRAINED_MODEL_EPOCH]

Arguments:
-h, --help - Show this help message and exit
--task - Task type for testing. Options: detection, rayleigh. Default is 'rayleigh'
--test-path - Path to directory containing images for testing
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
```

Example:

```
ACC=4

TEST_PATH=../synthetic_data_generation/examples/img_w_signal/
TRAINED_MODEL_PATH=./trained_model/mri_cnn_io_acc_${ACC}_hvd/hvd_cpts/
#TRAINED_MODEL_PATH=./trained_model/mri_cnn_io_acc_${ACC}_unet_hvd/hvd_cpts/

# Transfer-learned models
EPOCH=50

# Based model
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

* **Object generation using DDPM**

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

* **MR acquisition and reconstruction**

  This example shows forward projection and reconstruction of DDPM generated objects using the rSOS method to create test dataset. It saves the reconstructions in HDF5 format.

  Command-line Options:

```
Acceleration (int): Acceleration factor for sparse sampling (2, 4, 6, or 8).
```

  Usage:

```
python rsos_ddpm_test.py [acceleration factor]
```

  Examples:
    Run with acceleration factor 4:

```
python rsos_ddpm_test.py 4
```

* **Syntheric defect insertion**

  This script inserts doulet and singlet signals into DDPM generated objects. It saves the objects with signals in HDF5 format.

Command-line Options:

```
Acceleration (int): Acceleration factor for sparse sampling (2, 4, 6, or 8).
```

Usage:

```
python signal_insertion_test.py [acceleration factor]
```

Examples:
    Run with acceleration factor 4:

```
python signal_insertion_test.py 4
```

* **Statistical anslysis**

Statistical analysis includes two example scripts: 1) sample size determination via a power analysis and 2) statistical analysis for a pivotal study. Pre-installation of the iMRMC application is **NOT** recommended for the use of these scripts.

1. *Sample size determination*

   This script conducts a power analysis for sample size determination in our paper. To run the script, simply execute `power_analysis_BDG.R`. To use your own pilot data, please replace `pliot_data.csv` with your data, following the same format in `pliot_data.csv`, and update proportion correct by DLMO and its variance in the `power_analysis_BDG.R`.

2. *Pivotal study*

    This script conducts a similarity test to investigate whether DLMO performs similarly to human readers within a pre-defined margin of 0.1 proportion correct. To run the script, simply execute `similarity_test.R`. To use it for your own project, please update `DLMO reading results` section in `similarity_test.R`, and provide reading scores in the `reading_scores` folder following the same format.


## License and Copyright
DLMO is distributed under the MIT license. See [LICENSE](https://github.com/DIDSR/DLMO/blob/ec93f4b73e1caa5a21d89b0a41cd0a8681197999/LICENSE) for more information.

## Citation

## Disclaimer
