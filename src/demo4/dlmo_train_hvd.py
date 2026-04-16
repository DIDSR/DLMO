# Train the deep learning-based model observer.
# It supports distributed training using Horovod
# and handles various configurations through command-line arguments.
#
# Main components:
# 1. Argument parsing and setup
# 2. Data loading (training and validation)
# 3. Model initialization and optimization setup
# 4. Training loop with validation
# 5. Checkpoint saving and logging
#
#
# To use, source the following environment.
# source /anaconda3/base_env.sh
# conda activate dlmo # (with horovod build)
import torch
import numpy as np
import argparse
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed
import horovod.torch as hvd
import torch.nn.functional as F
from models import Net_7conv2_dropout as Net

import os
import math
import sys

from dataset import DatasetFromHdf5
from sklearn.metrics import roc_auc_score
import scipy.io
import utils
from tqdm import tqdm

# --------------------------------------Some basic settings ---------------------------------------#
parser = argparse.ArgumentParser(description='Train the DLMO.')
parser.add_argument('--acceleration', help='Acceleration factor ([1,2,4,6,8,10,12]).', type=int)
parser.add_argument('--train-data-path', help='Path to reconstructed MR images (with signals) for training DLMO.', type=str)
parser.add_argument('--val-data-path', help='Path to few samples of reconstructed MR images \
                    (with signals) for tuning DLMO.')
parser.add_argument('--output-path', type=str, default='trained_model', \
                    help='Path to save checkpoints and log files')
parser.add_argument('--pretrained-model-path', help='Transfered learning based on a previous \
                    trained model (provide path).')
parser.add_argument('--pretrained-model-checkpoint-format', default='checkpoint-{epoch}.pth.tar',
                    help='checkpoint file format')
parser.add_argument('--pretrained-model-epoch', type=int, default=170, help='Transfered learning \
                    based on a previous trained model (provide epoch).')
parser.add_argument('--nepochs', type=int, default=50, help='number of epochs to train')
parser.add_argument('--batch-size', help='Training batch size.', type=int)
parser.add_argument('--val-batch-size', type=int, default=16, help='batch size for validation/tuning data.')
parser.add_argument('--wd', type=float, default=0.0,
                    help='weight decay a.k.a regularization on weights')
parser.add_argument('--shuffle_patches', action="store_true", \
                    help="shuffles the train/validation patch pairs(input-target) at \
                    utils.data.DataLoader & not at the HDF5dataloader")
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--checkpoint-format', default='checkpoint-{epoch}.pth.tar',
                    help='checkpoint file format')
parser.add_argument('--save-log-ckpts', action="store_true", help="saves log writer and checkpoints")
parser.add_argument('--log-file-format', default='log-{epoch}.pkl',
                    help='log file format')

args = parser.parse_args()

acceleration = args.acceleration  # acceleration factor
if acceleration not in [1, 2, 4, 6, 8]:
    print('Invalid acceleration factor.', flush=True)
    sys.exit()
# CMD variables ---------------------------------------------
batch_size             = args.batch_size # training batch size 
val_batch_size         = args.val_batch_size # validation batch size 
batches_per_allreduce  = 1 # to implement other values, refer to https://github.com/prabhatkc/ct-recon/blob/main/Denoising/DLdenoise/main_hvd.py#L275
allreduce_batch_size   = batch_size * batches_per_allreduce
max_epochs             = args.nepochs
epoch_start            = 0

# Paths to i/o files --------------------------------------
train_data_path        = args.train_data_path   
val_data_path          = args.val_data_path
args.output_path       = args.output_path + str(acceleration) + "_hvd/"
output_path            = args.output_path 
pretrained_model_path  = args.pretrained_model_path
pretrained_model_epoch = args.pretrained_model_epoch

cmpr_dtype = 'float32'
if cmpr_dtype   == 'float16':
    torch_dtype = torch.float16
else:
    torch_dtype = torch.float32

if args.save_log_ckpts:
    # creating the main output dir 
    if not os.path.isdir(output_path): os.makedirs(output_path, exist_ok=True)
    # declaring the checkpoint fname
    checkpoint_folder = output_path + 'hvd_cpts/'
    if not os.path.isdir(checkpoint_folder): os.makedirs(checkpoint_folder, exist_ok=True)
    args.checkpoint_format = os.path.join(checkpoint_folder, args.checkpoint_format)
    # declaring the log fname
    current_log_folder = output_path + 'hvd_log/'
    if not os.path.isdir(current_log_folder): os.makedirs(current_log_folder, exist_ok=True)
    args.log_file_format = os.path.join(current_log_folder, args.log_file_format)

hvd.init()

# ------------------------------------- CUDA for PyTorch ------------------------------------------#
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
if use_cuda:
    # https://pytorch.org/docs/stable/notes/numerical_accuracy.html
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True  # allow fp16 compression
    torch.backends.cuda.matmul.allow_tf32 = True  # allow TF32 compression for faster calculation
    # Horovod: pin GPU to local rank.
    torch.cuda.set_device(hvd.local_rank())
cudnn.benchmark = True

# Display command line arguments
if hvd.rank() == 0:
  print('\n----------------------------------------')
  print('Command line arguements')
  print('----------------------------------------')
  print("\nNo. of gpus used:", hvd.size())
  for i in args.__dict__: print((i), ':', args.__dict__[i])
  print('\n----------------------------------------\n')

# Horovod: print logs on the first worker
verbose = 1 if hvd.rank() == 0 else 0

# ==================================================================
# load training data
# ==================================================================
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

train_dataset = DatasetFromHdf5(hvd=hvd, file_path=train_data_path, \
                                mod_num=hvd.size() * batch_size)

train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, \
                num_replicas=hvd.size(), rank=hvd.rank(), shuffle=args.shuffle_patches)
train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=allreduce_batch_size, \
               sampler=train_sampler, **kwargs)
tr_tot_size   = train_dataset.data.shape[0]
dim1          = train_dataset.data.shape[2]
dim2          = train_dataset.data.shape[3]

if hvd.rank() == 0: print(" Dimension of training (input <-> target) batches is: {} <-> {}"\
                        .format(train_dataset.data.shape, train_dataset.target.shape))
    
# ==================================================================
# load validation data
# ==================================================================
val_dataset   = DatasetFromHdf5(hvd=hvd, file_path=val_data_path, \
                              mod_num=hvd.size() * val_batch_size)
val_sampler   = torch.utils.data.distributed.DistributedSampler(
                val_dataset, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=args.shuffle_patches)
val_loader    = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size,
                                         sampler=val_sampler, **kwargs)
val_tot_size  = val_dataset.data.shape[0]

if hvd.rank() == 0: print(" Dimension of validation (input <-> target) batches is: {} <-> {}"\
                            .format(val_dataset.data.shape, val_dataset.target.shape))
# ==================================================================
# Initializing Model and objective loss function
# ==================================================================
if hvd.rank() == 0: print("Building model .... ")
net = Net(dim1=dim1, dim2=dim2, filter_size=7)

# transfer models to cuda
if use_cuda: model = net.cuda()

optimizer   = optim.Adam(model.parameters(), \
              lr=(1e-5 * batches_per_allreduce * hvd.size()), \
              weight_decay=args.wd)

# Horovod: (optional) compression algorithm.
compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

# Horovod: wrap optimizer with DistributedOptimizer.
optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters(), \
  compression=compression, \
  backward_passes_per_step=batches_per_allreduce)

criterion = nn.BCEWithLogitsLoss()

# Restore from a previous checkpoint, if initial_epoch is specified.
# Horovod: restore on the first worker which will broadcast weights to other workers.
if pretrained_model_path and hvd.rank() == 0:
    filepath = pretrained_model_path + args.pretrained_model_checkpoint_format.format(epoch=pretrained_model_epoch)
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model'])
    if acceleration == 1:
        optimizer.load_state_dict(checkpoint['optimizer'])
    print("Loaded pretrained network from:", filepath, flush=True)

# Horovod: broadcast parameters & optimizer state.
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

# ==================================================================
# train the model
# ==================================================================
if hvd.rank() == 0: print("Training .... ")

train_acc_arr = np.empty((max_epochs, 1))
val_acc_arr   = np.empty((max_epochs, 1))
train_loss_arr= np.empty((max_epochs, 1))
val_loss_arr  = np.empty((max_epochs, 1))

def train(epoch):
    model.train()
    train_loss = utils.Metric('train_loss')
    train_acc  = utils.Metric('train_acc')
    with tqdm(total=len(train_loader),
            desc='Epoch  #{}'.format(epoch + 1),
            disable=not verbose) as t:
        for batch_idx, (data, target) in enumerate(train_loader):
            if use_cuda: 
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output  = model(data)
            loss    = criterion(output, target)
            loss.backward()
            train_loss.update(loss, hvd)
            roc_acc = torch.tensor(roc_auc_score(np.squeeze(target.cpu().detach().numpy()), np.squeeze(output.cpu().detach().numpy())))
            train_acc.update(roc_acc, hvd)
            optimizer.step()

            t.set_postfix({'train acc': train_acc.avg.item(), 'train loss': train_loss.avg.item()})
            t.update(1)
    if args.save_log_ckpts:
        train_acc_arr[epoch]  = train_acc.avg.item()
        train_loss_arr[epoch] = train_loss.avg.item()

def validate(epoch):
    model.eval()
    val_loss = utils.Metric('val_loss')
    val_acc  = utils.Metric('val_acc')
    with tqdm(total=len(val_loader),
            desc='Epoch  #{}'.format(epoch + 1),
            disable=not verbose) as t:
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                if use_cuda: 
                    data, target = data.cuda(), target.cuda()
                output  = model(data)
                loss    = criterion(output, target)
                val_loss.update(loss, hvd)
                roc_acc = torch.tensor(roc_auc_score(np.squeeze(target.cpu().detach().numpy()), np.squeeze(output.cpu().detach().numpy())))
                val_acc.update(roc_acc, hvd)
                t.set_postfix({'val acc': val_acc.avg.item(), 'val loss': val_loss.avg.item()})
                t.update(1)
    if args.save_log_ckpts:
        val_acc_arr[epoch]  = val_acc.avg.item()
        val_loss_arr[epoch] = val_loss.avg.item()

# ====================================================================
# Main function with train, validate; save weights and metrics
# ======================================================================
for epoch in range(epoch_start, max_epochs):
    train(epoch)
    validate(epoch)
    # save trained weights 
    if (args.save_log_ckpts): utils.save_checkpoint(epoch, args, hvd.rank(), model, optimizer)

# saving per epoch metrics into an h5 file 
if (args.save_log_ckpts):
    loss_acc = {'train acc' : train_acc_arr[train_acc_arr != 0], \
                'val acc'   : val_acc_arr[val_acc_arr != 0], \
                'train loss': train_loss_arr[train_loss_arr != 0], \
                'val loss'  : val_loss_arr[val_loss_arr != 0]}
    utils.save_log_to_h5(args, hvd.rank(), loss_acc)