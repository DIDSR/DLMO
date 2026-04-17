# DLMO training

The deep learning–based model observer (DLMO) training code provided here supports distributed training using Horovod, as well as serial training without Horovod (using 1 GPU). It also accommodates various configurations through command-line arguments.

The usage demo below runs 80 image-based DLMO training using rSOS reconstructions at an acceleration factor of 4, obtained from the previous Demo 3. This example is intended to demonstrate a dummy convergence of the DLMO training using pretrained weights and to show that the training code works.

Note that in our [DLMO paper](https://arxiv.org/abs/2602.22535), the total number of training MR images for both the base model (i.e., acceleration = 1) and the refined DLMOs for each acceleration factor (i.e., 4× and 8×) was 160,000. To re-obtain the backbone-trained model (for acc = 1) and the retuned model (for acc > 1) provided in this repository and discussed in our DLMO paper, you will need to use training based on 160k images. All these images can be generated, have signals inserted, and be reconstructed by following the instructions provided in Demo 1, Demo 2, and Demo 3, respectively.

Also note that the pretrained weights provided in this repository were obtained using the Horovod-based training script (`train_dlmo_hvd.py`). However, you can also train from scratch or fine-tune the model without requiring the Horovod package by using `train_dlmo.py`.

Main components:

  1. Argument parsing and setup
  2. Data loading (training and validation)
  3. Model initialization and optimization setup
  4. Training and tuning
  5. Checkpoint saving and logging

Usage (without horovod build):
```
python dlmo_train.py [-h] [--acceleration ACCELERATION] [--train-data-path TRAIN_DATA_PATH] [--val-data-path VAL_DATA_PATH]
                    [--output-path OUTPUT_PATH] [--pretrained-model-path PRETRAINED_MODEL_PATH]
                    [--pretrained-model-checkpoint-format PRETRAINED_MODEL_CHECKPOINT_FORMAT]
                    [--pretrained-model-epoch PRETRAINED_MODEL_EPOCH] [--nepochs NEPOCHS] --batch-size BATCH_SIZE
                    [--val-batch-size VAL_BATCH_SIZE] [--wd WD] [--shuffle_patches] [--fp16-allreduce]
                    [--checkpoint-format CHECKPOINT_FORMAT] [--save-log-ckpts] [--log-file-format LOG_FILE_FORMAT]
Train the DLMO.

optional arguments:
  -h, --help  show this help message and exit
  --acceleration                        Acceleration factor ([1,2,4,6,8,10,12]).
  --train-data-path                     Path to reconstructed MR images (with signals) for
                                        training DLMO.
  --val-data-path                       Path to few samples of reconstructed MR images (with
                                        signals) for tuning DLMO.
  --output-path                         Path to save checkpoints and log files
  --pretrained-model-path               Transfered learning based on a previous trained model
                                        (provide path).
  --pretrained-model-checkpoint-format  checkpoint file format
  --pretrained-model-epoch              Transfered learning based on a previous trained model
                                        (provide epoch).
  --nepochs                             number of epochs to train
  --batch-size                          Training Batch size.
  --val-batch-size                      input batch size for the validation/tuning data.
  --wd                                  weight decay a.k.a regularization on weights
  --shuffle_patches                     Shuffles train/validation patch pairs at DataLoader level.
  --fp16-allreduce                      Retained for CLI compatibility; ignored without Horovod.
  --checkpoint-format                   checkpoint file format
  --save-log-ckpts                      saves log writer and checkpoints
  --log-file-format                     log file format
```

Non-horovod Example: Training models with transfer learning on accelerated data:

```
# conda activate dlmo # WITHOUT horovod build 

PRETRAIN_EPOCH=170
ACC=4
NGPUS=1
NEPOCH=35 #dummy value

PRETRAIN_PATH=../demo5/DLMO_test/trained_model/mri_cnn_dlmo_acc_1_hvd/
TRAIN_DATA_PATH=../demo3/rsos_rec/test_acc4_rsos.hdf5
VAL_DATA_PATH=../demo3/rsos_rec/test_acc4_rsos.hdf5
OUTPUT_FLD_PATH=trained_model/mri_cnn_dlmo_acc_

time python dlmo_train.py \
--acceleration $ACC \
--nepochs $NEPOCH  \
--train-data-path $TRAIN_DATA_PATH \
--val-data-path $VAL_DATA_PATH \
--output-path $OUTPUT_FLD_PATH \
--batch-size 20 \
--val-batch-size 20 \
--shuffle_patches \
--pretrained-model-path ${PRETRAIN_PATH} \
--pretrained-model-epoch ${PRETRAIN_EPOCH} \
--save-log-ckpts \
--log-file-format log_${NGPUS}_gpus.hdf5
```

Non-horovod Example: Training the base model at acceleration=1:

```
# conda activate dlmo # WITHOUT horovod build 

ACC=1
NGPUS=1
NEPOCH=120 #dummy value
TRAIN_DATA_PATH=../demo3/rsos_rec/test_acc4_at_acc1_rsos.hdf5
VAL_DATA_PATH=../demo3/rsos_rec/test_acc4_at_acc1_rsos.hdf5
OUTPUT_FLD_PATH=trained_model/mri_cnn_dlmo_acc_

time python dlmo_train.py \
--acceleration $ACC \
--nepochs $NEPOCH \
--train-data-path $TRAIN_DATA_PATH \
--val-data-path $VAL_DATA_PATH \
--output-path $OUTPUT_FLD_PATH \
--batch-size 20 \
--val-batch-size 20 \
--shuffle_patches \
--save-log-ckpts \
--log-file-format log_${NGPUS}_gpus.hdf5
```

Usage (with horovod build):
```
horovodrun -np $NGPUS -H localhost:$NGPUS python dlmo_train_hvd.py [-h] 
                         [--acceleration ACCELERATION] [--train-data-path TRAIN_DATA_PATH]
                         [--val-data-path VAL_DATA_PATH] [--output-path OUTPUT_PATH]
                         [--pretrained-model-path PRETRAINED_MODEL_PATH]
                         [--pretrained-model-checkpoint-format PRETRAINED_MODEL_CHECKPOINT_FORMAT]
                         [--pretrained-model-epoch PRETRAINED_MODEL_EPOCH] [--nepochs NEPOCHS] 
                         [--batch-size BATCH_SIZE] [--val-batch-size VAL_BATCH_SIZE]
                         [--shuffle_patches] 
                         [--fp16-allreduce] [--checkpoint-format CHECKPOINT_FORMAT]
                         [--save-log-ckpts] [--log-file-format LOG_FILE_FORMAT]

Train the DLMO.

optional arguments:
  -h, --help  show this help message and exit
  --acceleration                        Acceleration factor ([1,2,4,6,8,10,12]).
  --train-data-path                     Path to reconstructed MR images (with signals) for
                                        training DLMO.
  --val-data-path                       Path to few samples of reconstructed MR images (with
                                        signals) for tuning DLMO.
  --output-path                         Path to save checkpoints and log files
  --pretrained-model-path               Transfered learning based on a previous trained model
                                        (provide path).
  --pretrained-model-checkpoint-format  checkpoint file format
  --pretrained-model-epoch              Transfered learning based on a previous trained model
                                        (provide epoch).
  --nepochs                             number of epochs to train
  --batch-size                          Training Batch size.
  --val-batch-size                      input batch size for the validation/tuning data.
  --wd                                  weight decay a.k.a regularization on weights
  --shuffle_patches                     shuffles the train/validation patch pairs(input-
                                        target) at utils.data.DataLoader & not at the
                                        HDF5dataloader
  --fp16-allreduce                      use fp16 compression during allreduce
  --checkpoint-format                   checkpoint file format
  --save-log-ckpts                      saves log writer and checkpoints
  --log-file-format                     log file format
```

Horovod Example: Training models with transfer learning using accelerated data:

```
# conda activate dlmo # with horovod build  and trained using A100 GPUS

PRETRAIN_EPOCH=170
ACC=4
NGPUS=2
NEPOCH=50 #dummy value

PRETRAIN_PATH=../demo5/DLMO_test/trained_model/mri_cnn_dlmo_acc_1_hvd/
TRAIN_DATA_PATH=../demo3/rsos_rec/test_acc4_rsos.hdf5
VAL_DATA_PATH=../demo3/rsos_rec/test_acc4_rsos.hdf5
OUTPUT_FLD_PATH=trained_model/mri_cnn_dlmo_acc_

time NCCL_DEBUG=INFO horovodrun -np $NGPUS -H localhost:$NGPUS python dlmo_train_hvd.py \
--acceleration $ACC \
--nepochs $NEPOCH  \
--train-data-path $TRAIN_DATA_PATH \
--val-data-path $VAL_DATA_PATH \
--output-path $OUTPUT_FLD_PATH \
--batch-size 10 \
--val-batch-size 10 \
--shuffle_patches \
--pretrained-model-path ${PRETRAIN_PATH} \
--pretrained-model-epoch ${PRETRAIN_EPOCH} \
--save-log-ckpts \
--log-file-format log_${NGPUS}_gpus.hdf5
```

Horovod Example: Training the base model at the fully-sampled data (at acceleration=1):

```
# conda activate dlmo # with horovod build and trained using A100 GPUS

ACC=1
NGPUS=2
NEPOCH=120 #dummy value
TRAIN_DATA_PATH=../demo3/rsos_rec/test_acc4_at_acc1_rsos.hdf5
VAL_DATA_PATH=../demo3/rsos_rec/test_acc4_at_acc1_rsos.hdf5
OUTPUT_FLD_PATH=trained_model/mri_cnn_dlmo_acc_

time horovodrun -np $NGPUS -H localhost:$NGPUS python dlmo_train_hvd.py \
--acceleration $ACC \
--nepochs $NEPOCH \
--train-data-path $TRAIN_DATA_PATH \
--val-data-path $VAL_DATA_PATH \
--output-path $OUTPUT_FLD_PATH \
--batch-size 20 \
--val-batch-size 20 \
--shuffle_patches \
--save-log-ckpts \
--log-file-format log_${NGPUS}_gpus.hdf5
```

