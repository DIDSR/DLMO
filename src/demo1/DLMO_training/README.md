# DLMO training

Train the deep learning-based model observer. It supports distributed training using Horovod and handles various configurations through command-line arguments. The number of training samples for both the base model and the refined DLMOs for each acceleration factor was 168,000. Due to this large number  of training samples, only scripts are provided here.

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