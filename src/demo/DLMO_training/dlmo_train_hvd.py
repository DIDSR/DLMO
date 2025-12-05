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
# source /home/prabhat.kc/anaconda3/base_env.sh
# source /home/prabhat.kc/anaconda3/horovod_sm80_env.sh

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
parser = argparse.ArgumentParser(description='Train the CNN IO.')
parser.add_argument('--task', help='Task type (detection/rayleigh).')
parser.add_argument('--acceleration', help='Acceleration factor ([1,2,4,6,8,10,12]).', type=int)
parser.add_argument('--pretrained-model-path', help='Transfered learning based on a previous trained model (provide path).')
parser.add_argument('--pretrained-model-checkpoint-format', default='checkpoint-{epoch}.pth.tar',
                    help='checkpoint file format')
parser.add_argument('--pretrained-model-epoch', type=int, default=150, help='Transfered learning based on a previous trained model (provide epoch).')
parser.add_argument('--batch-size', help='Batch size.', type=int)
parser.add_argument('--val-batch-size', type=int, default=16, help='input batch size for validation data.')
parser.add_argument('--batches-per-allreduce', type=int, default=1,
                    help='number of batches processed locally before executing allreduce across workers;'
                    'It multiplies the total batch size. (RHR: 1 loss function eqs 1 batches-per-allreduce')
parser.add_argument('--shuffle_patches', action="store_true", \
                    help="shuffles the train/validation patch pairs(input-target) at \
                                                                    utils.data.DataLoader & not at the HDF5dataloader")
parser.add_argument('--wd', type=float, default=0.0,
                    help='weight decay a.k.a regularization on weights')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--checkpoint-format', default='checkpoint-{epoch}.pth.tar',
                    help='checkpoint file format')
parser.add_argument('--save-log-ckpts', action="store_true", help="saves log writer and checkpoints")
parser.add_argument('--log-file-format', default='log-{epoch}.pkl',
                    help='log file format')

args = parser.parse_args()

task_type = args.task  # task type (detection/rayleigh)
if task_type not in ['detection', 'rayleigh']:
    print('Invalid task type.', flush=True)
    sys.exit()

acceleration = args.acceleration  # acceleration factor
if acceleration not in [1, 2, 4, 6, 8]:
    print('Invalid acceleration factor.', flush=True)
    sys.exit()

batch_size = args.batch_size
val_batch_size = args.val_batch_size
batches_per_allreduce = args.batches_per_allreduce
allreduce_batch_size = batch_size * batches_per_allreduce

pretrained_model_path = args.pretrained_model_path
pretrained_model_epoch = args.pretrained_model_epoch

epoch_start = 0
max_epochs = 200

train_data_path = "/projects01/didsr-aiml/zitong.yu/DLMO/data/train/"
val_data_path = "/projects01/didsr-aiml/zitong.yu/DLMO/data/test/"

output_path = "/projects01/didsr-aiml/zitong.yu/DLMO/src/dlmo/trained_model/mri_" + task_type + "_acc_" + str(acceleration) + "_hvd/"
if not os.path.isdir(output_path): os.makedirs(output_path, exist_ok=True)

dim1, dim2 = 260, 311
n_std = 15  # # is the always 15 for all acceleration factors ? KL: Yes
n_coil = 8
cmpr_dtype = 'float32'

te_half_size = 4000
te_tot_size = 2 * te_half_size

tr_tot_size = 160000

if te_tot_size % batch_size != 0:
    print("Batch size should divide the total testing size.")
    sys.exit()
elif tr_tot_size % batch_size != 0:
    print("Batch size should divide the total training size.")
    sys.exit()

if cmpr_dtype == 'float16':
    torch_dtype = torch.float16
else:
    torch_dtype = torch.float32

if args.save_log_ckpts:
    # declaring the checkpoint fname
    checkpoint_folder = output_path + '/hvd_cpts/'
    if not os.path.isdir(checkpoint_folder): os.makedirs(checkpoint_folder, exist_ok=True)
    args.checkpoint_format = os.path.join(checkpoint_folder, args.checkpoint_format)
    # declaring the log fname
    current_log_folder = output_path + '/hvd_log/'
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

train_dataset = DatasetFromHdf5(hvd=hvd, file_path=train_data_path + "train_acc" + str(acceleration) + "_rsos.hdf5", \
                                mod_num=hvd.size() * batch_size)

train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, \
  num_replicas=hvd.size(), rank=hvd.rank(), shuffle=args.shuffle_patches)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=allreduce_batch_size, \
  sampler=train_sampler, **kwargs)
if hvd.rank() == 0: print(" Dimension of training (input <-> target) batches is: {} <-> {}".format(train_dataset.data.shape, \
  train_dataset.target.shape))

# ==================================================================
# load validation data
# ==================================================================
val_dataset = DatasetFromHdf5(hvd=hvd, file_path=val_data_path + "val_acc" + str(acceleration) + "_rsos.hdf5", \
                              mod_num=hvd.size() * val_batch_size)
val_sampler = torch.utils.data.distributed.DistributedSampler(
    val_dataset, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=args.shuffle_patches)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size,
                                         sampler=val_sampler, **kwargs)
if hvd.rank() == 0: print(" Dimension of validation (input <-> target) batches is: {} <-> {}".format(val_dataset.data.shape, \
  val_dataset.target.shape))

# ==================================================================
# Initializing Model and objective loss function
# ==================================================================
if hvd.rank() == 0: print("Building model .... ")
net = Net(dim1=dim1, dim2=dim2, filter_size=7)

# transfer models to cuda
if use_cuda:
  model = net.cuda()

optimizer = optim.Adam(model.parameters(), \
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
train_loss = utils.Metric('train_loss')
val_loss = utils.Metric('val_loss')
train_acc_his = np.empty((max_epochs, 1))
val_acc_his = np.empty((max_epochs, 1))
train_loss_his = np.empty((max_epochs, 1))
val_loss_his = np.empty((max_epochs, 1))
for epoch in range(epoch_start, max_epochs):
    # Training
    with tqdm(total=len(train_loader),
            desc='Epoch  #{}'.format(epoch + 1),
            disable=not verbose) as t:
        for batch_idx, (data, target) in enumerate(train_loader):

            if use_cuda: data, target = data.cuda(), target.cuda()

            model.train()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            train_loss.update(loss, hvd)

            cur_train_acc = roc_auc_score(target.cpu().numpy(), output.cpu().detach().numpy())

            optimizer.step()

            # Validation step (every 50 batches)
            if (batch_idx + 1) % 50 == 0:
                with torch.no_grad():
                    val_outputs_flag = 0
                    for data, target in val_loader:
                        if use_cuda:
                            data, target = data.cuda(), target.cuda()

                        model.eval()
                        output = model(data)
                        if val_outputs_flag == 0:
                            val_outputs = output
                            val_targets = target
                            val_outputs_flag = 1
                        elif val_outputs_flag == 1:
                            val_outputs = torch.cat((val_outputs, output), 0)
                            val_targets = torch.cat((val_targets, target), 0)
                        loss = criterion(output, target)
                        val_loss.update(loss, hvd)
                cur_val_acc = roc_auc_score(val_targets.cpu().numpy(), val_outputs.cpu().numpy())
                t.set_postfix({'train acc': cur_train_acc, \
                               'val acc': cur_val_acc, \
                               'train loss': train_loss.avg.item()})
                t.update(50)

    # Save checkpoints and logs if enabled
    if (args.save_log_ckpts):
        utils.save_checkpoint(epoch, args, hvd, model, optimizer)
        train_acc_his[epoch] = cur_train_acc
        val_acc_his[epoch] = cur_val_acc
        train_loss_his[epoch] = train_loss.avg.item()
        val_loss_his[epoch] = val_loss.avg.item()
        loss_acc = {'train acc': train_acc_his[train_acc_his != 0], \
                    'val acc': val_acc_his[val_acc_his != 0], \
                    'train loss': train_loss_his[train_loss_his != 0], \
                    'val loss': val_loss_his[val_loss_his != 0]}
        utils.save_log_to_h5(args, hvd, loss_acc)
