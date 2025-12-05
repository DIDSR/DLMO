# Trains a CNN denoiser for MRI reconstruction tasks, supporting various model architectures
# and acceleration factors. It uses Horovod for distributed training across multiple GPUs.
# Command-line Options:
#     --task: Task type (detection/rayleigh). Default is 'detection'.
#     --training-path: Path to the objects for training.
#     --val-path: Path to the objects for validation.
#     --acceleration: Acceleration factor (2, 4, 6, or 8). Default is 2.
#     --model_name: CNN denoiser model (cnn3, redcnn, udncnn, dncnn, unet). Default is 'unet'.
#     --num-channels: Number of input channels (1 for grayscale, 3 for RGB). Default is 1.
#     --batch-size: Batch size for training.
#     --max-epochs: Maximum number of epochs for training.
#     --val-batch-size: Batch size for validation. Default is 16.
#     --batches-per-allreduce: Number of batches processed locally before allreduce. Default is 1.
#     --shuffle_patches: Flag to shuffle patches during data loading.
#     --fp16-allreduce: Use fp16 compression during allreduce.
#     --save-log-ckpts: Flag to save logs and checkpoints.
#     --checkpoint-format: Format for checkpoint files. Default is 'checkpoint-{epoch}.pth.tar'.
#     --log-file-format: Format for log files. Default is 'log-{epoch}.pkl'.
#     --pretrained-model-path: Path to a pre-trained model for transfer learning.
#     --pretrained-model-epoch: Epoch number of the pre-trained model to use. Default is 150.
#     --wd: Weight decay for regularization. Default is 0.0.
#     --base-lr: Base learning rate. Default is 1e-5.
#     --seed: Random seed for reproducibility. Default is 37.
#
#
# Note: training data (accelerated imgs, fully-sampled imgs) generated on-the-fly
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
import models

import pytorch_msssim

import os
import math
import sys

from dataset import list_all_npy_files
import scipy.io
import utils
from tqdm import tqdm
import add_signals

# from torchsummary import summary

import time

start_time = time.time()
# --------------------------------------Some basic settings ---------------------------------------#
parser = argparse.ArgumentParser(description='Train the CNN denoiser.')
parser.add_argument('--task', default='detection', help='Task type (detection/rayleigh).')
parser.add_argument('--training-path', \
                    help='Path to the objects for training.')
parser.add_argument('--val-path', \
                    help='Path to the objects for validation.')
parser.add_argument('--acceleration', type=int, default=2, \
                    help='Acceleration factor in range of 2 to 12.')
parser.add_argument('--model_name', default='unet', help='CNN denoiser model.')
parser.add_argument('--num-channels', type=int, default=1, help='3 for rgb images and 1 for gray scale images')
parser.add_argument('--batch-size', help='Batch size.', type=int)
parser.add_argument('--max-epochs', help='Max epoch number.', type=int)
parser.add_argument('--val-batch-size', type=int, default=16, help='input batch size for validation data.')
parser.add_argument('--batches-per-allreduce', type=int, default=1,
                    help='number of batches processed locally before executing allreduce across workers;'
                    'It multiplies the total batch size. (RHR: 1 loss function eqs 1 batches-per-allreduce')
parser.add_argument('--shuffle_patches', action="store_true", \
                    help="shuffles the train/validation patch pairs(input-target) at \
                          utils.data.DataLoader & not at the HDF5dataloader")
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--save-log-ckpts', action="store_true", help="saves log writer and checkpoints")
parser.add_argument('--checkpoint-format', default='checkpoint-{epoch}.pth.tar',
                    help='checkpoint file format')
parser.add_argument('--log-file-format', default='log-{epoch}.pkl',
                    help='log file format')
parser.add_argument('--pretrained-model-path', help='Transfered learning based on a previous trained model (provide path).')
parser.add_argument('--pretrained-model-checkpoint-format', default='checkpoint-{epoch}.pth.tar',
                    help='checkpoint file format')
parser.add_argument('--pretrained-model-epoch', type=int, default=150, help='Transfered learning based on a previous trained model (provide epoch).')
parser.add_argument('--wd', type=float, default=0.0,
                    help='weight decay a.k.a regularization on weights')
parser.add_argument('--base-lr', type=float, default=1e-5,
                    help='Base learning rate')
parser.add_argument('--seed', type=int, default=37,
                    help='random seed')

args = parser.parse_args()

task_type = args.task  # task type (detection/rayleigh)
if task_type not in ['detection', 'rayleigh']:
    print('Invalid task type.', flush=True)
    sys.exit()

acceleration = args.acceleration  # acceleration factor
if acceleration not in [2, 4, 6, 8]:
    print('Invalid acceleration factor.', flush=True)
    sys.exit()

# ----------------------------Signal information------------------------------------------------------------#

A_dict = {'1':0.3, \
          '2':0.3, \
          '4':0.7, \
          '6':1.0, \
          '8':1.3}
A = A_dict[str(acceleration)]

wid = 1.75
L = [4, 5, 6, 7, 8, 9]


loc = np.load("/projects01/didsr-aiml/zitong.yu/DLMO/src/mri_loc.npy")

# ----------------------------Training settings------------------------------------------------------------#

model_name = args.model_name
num_channels = args.num_channels

batch_size = args.batch_size
val_batch_size = args.val_batch_size
batches_per_allreduce = args.batches_per_allreduce
allreduce_batch_size = batch_size * batches_per_allreduce
base_lr = args.base_lr

pretrained_model_path = args.pretrained_model_path
pretrained_model_epoch = args.pretrained_model_epoch

epoch_start = 0
max_epochs = args.max_epochs

train_data_path = args.training_path
val_data_path = args.val_path

output_path = "/projects01/didsr-aiml/zitong.yu/DLMO/src/ai_rec/trained_model/mri_" + model_name + "_" + task_type + "_acc_" + str(acceleration) + "/"
if not os.path.isdir(output_path): os.makedirs(output_path, exist_ok=True)

dim1, dim2 = 260, 311
pad = (4, 5, 6, 6) # zero-padding for unet training
n_std = 15  # is the always 15 for all acceleration factors ? KL: Yes
n_coil = 8
cmpr_dtype = 'float32'

val_half_size = 2000
val_tot_size = 2 * val_half_size

train_tot_size = 16000

if cmpr_dtype == 'float16':
    torch_dtype = torch.float16
else:
    torch_dtype = torch.float32

hvd.init()

if args.save_log_ckpts:
    # declaring the checkpoint fname
    checkpoint_folder = output_path + '/hvd_cpts/'
    if not os.path.isdir(checkpoint_folder): os.makedirs(checkpoint_folder, exist_ok=True)
    args.checkpoint_format = os.path.join(checkpoint_folder, args.checkpoint_format)
    # declaring the log fname
    current_log_folder = output_path + '/hvd_log/'
    if not os.path.isdir(current_log_folder): os.makedirs(current_log_folder, exist_ok=True)
    args.log_file_format = os.path.join(current_log_folder, args.log_file_format)

# ------------------------------------- CUDA for PyTorch ------------------------------------------#
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(args.seed)

if use_cuda:
    # https://pytorch.org/docs/stable/notes/numerical_accuracy.html
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True  # allow fp16 compression
    torch.backends.cuda.matmul.allow_tf32 = True  # allow TF32 compression for faster calculation
    # Horovod: pin GPU to local rank.
    torch.cuda.set_device(hvd.local_rank())
    torch.cuda.manual_seed(args.seed)
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

# ----------------------------Load mask------------------------------------------------------------#

if acceleration > 1:
    if hvd.rank() == 0: print('\nAcceleration factor: ' + str(acceleration) + 'x\n', flush=True)
    # # Load mask
    mask_dir = "/projects01/didsr-aiml/zitong.yu/DLMO/src/masks/mask_Poisson_" + str(acceleration) + "_fold.npy"
    mask = np.load(mask_dir)  # shape of (260, 311)
    mask = np.reshape(mask, (1, 1, dim1, dim2))
    mask = torch.tensor(mask, dtype=torch.complex64)
    mask = mask.to(device)
    if hvd.rank() == 0: print("\nMask shape: ", mask.shape, "\n", flush=True)  # shape of [1, 1, 260, 311]

# ----------------------------Load sensitivity map-------------------------------------------------#
map_dir = "/projects01/didsr-aiml/zitong.yu/DLMO/src/sensitivity_8coils.npy"
sensi_map = np.load(map_dir)  # shaped (8, 260, 311)
sensi_map = np.reshape(sensi_map, (1, -1, dim1, dim2))  # shaped (1, 8, 260, 311)
sensi_map = torch.tensor(sensi_map, dtype=torch_dtype).to(device)
# sensi_map = sensi_map.to(device)
if hvd.rank() == 0: print('shape of the loaded sensitivity map: ', sensi_map.shape, 'and its dtype is', sensi_map.dtype, flush=True)

# ==================================================================
# Initializing Model and objective loss function
# ==================================================================
if hvd.rank() == 0: print("Building model .... \n")

if args.model_name == 'cnn3':
    from models import CNN3
    model = CNN3(num_channels=num_channels)
elif args.model_name == 'redcnn':
    from models import REDcnn10
    model = REDcnn10(idmaps=3)  # idmaps: no of skip connections (only 1 or 3 available)
elif args.model_name == 'udncnn':
    from models import UDnCNN
    model = UDnCNN(D=10)  # D is no of layers
elif args.model_name == 'dncnn':
    from models import DnCNN
    model = DnCNN(D=17, num_channels=num_channels)  # default: layers used is 17, bn=True, dropout=F
elif args.model_name == 'unet':
    from models import UNet
    model = UNet()  # UNet used in Kaiyan's SPIE paper
else:
    print("ERROR! Re-check DNN model (architecture) string!")
    sys.exit()

if hvd.rank() == 0: utils.print_model_summary(model, (1, dim1 + 12, dim2 + 9))

# transfer models to cuda
if use_cuda:
    model = model.cuda()

# ==================================================================
# Initializing the optimizer type
# ==================================================================

optimizer = optim.Adam(model.parameters(), \
                       lr=(base_lr * batches_per_allreduce * hvd.size()), \
                       weight_decay=args.wd)

# Horovod: (optional) compression algorithm.
compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

# Horovod: wrap optimizer with DistributedOptimizer.
optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters(), \
  compression=compression, \
  backward_passes_per_step=batches_per_allreduce)

mse_loss = nn.MSELoss()

# Restore from a previous checkpoint, if initial_epoch is specified.
# Horovod: restore on the first worker which will broadcast weights to other workers.
if pretrained_model_path and hvd.rank() == 0:
    filepath = pretrained_model_path + '/hvd_cpts/' + \
               args.pretrained_model_checkpoint_format.format(epoch=pretrained_model_epoch)
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("Loaded pretrained network from:", filepath, flush=True)

# Horovod: broadcast parameters & optimizer state.
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

# ==================================================================
# load training data
# ==================================================================
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

if hvd.rank() == 0: print("\nReading the training dataset ...", flush=True)

train_dataset = np.empty([0, dim1, dim2], dtype=cmpr_dtype)
train_dataset = np.append(train_dataset, list_all_npy_files(train_data_path + "test_out_52k/", cmpr_dtype=cmpr_dtype, unity_normalize=True), axis=0)  # these data are of type float64. We c

init_train_som = train_dataset.shape[0]

train_dataset = train_dataset[:train_tot_size,:,:]

if hvd.rank() == 0: print("\nAdding signals to the training dataset ...", flush=True)
train_dataset = add_signals.AddSignalRayleigh(train_dataset, A, wid, L, loc, train_tot_size / 2, train_tot_size, dim1, dim2)

train_dataset = np.reshape(train_dataset, (train_tot_size, 1, dim1, dim2))  # second index stores info on coil

train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, \
  num_replicas=hvd.size(), rank=hvd.rank(), shuffle=args.shuffle_patches)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=allreduce_batch_size, \
  sampler=train_sampler, **kwargs)
if hvd.rank() == 0:
    print('Out of %d SOMs, %d of them are used to train CNN denoiser\n' % (init_train_som, train_dataset.shape[0]), flush=True)  #
    print('min and max in training data: [%.4f, %.4f]' % (np.min(train_dataset), np.max(train_dataset)), flush=True)
    print('Shape of the loaded training data is:', train_dataset.shape, 'and its dtype is:', train_dataset.dtype, flush=True)

# ==================================================================
# load validation data
# ==================================================================

if hvd.rank() == 0: print("\nReading the validation dataset ...", flush=True)

val_dataset = np.empty([0, dim1, dim2], dtype=cmpr_dtype)
val_dataset = np.append(val_dataset, list_all_npy_files(val_data_path + "test_out_24k/", cmpr_dtype=cmpr_dtype, unity_normalize=True), axis=0)

init_val_som = val_dataset.shape[0]

val_dataset = val_dataset[:val_tot_size,:,:]

if hvd.rank() == 0: print("\nAdding signals to the validation dataset ...", flush=True)
val_dataset = add_signals.AddSignalRayleigh(val_dataset, A, wid, L, loc, val_half_size, val_tot_size, dim1, dim2)

val_dataset = np.reshape(val_dataset, (val_tot_size, 1, dim1, dim2))  # second index stores info on coil

val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, \
  num_replicas=hvd.size(), rank=hvd.rank(), shuffle=args.shuffle_patches)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=allreduce_batch_size, \
  sampler=val_sampler, **kwargs)
if hvd.rank() == 0:
    print('Out of %d SOMs, %d of them are used to validate CNN denoiser\n' % (init_val_som, val_dataset.shape[0]), flush=True)  #
    print('min and max in validation data: [%.4f, %.4f]' % (np.min(val_dataset), np.max(val_dataset)), flush=True)
    print('Shape of the loaded validation data is:', val_dataset.shape, 'and its dtype is:', val_dataset.dtype, flush=True)

# ==================================================================
# train the model
# ==================================================================
if hvd.rank() == 0: print("Training .... ")
train_loss_his = np.zeros((max_epochs, 1))
val_loss_his = np.zeros((max_epochs, 1))
train_ssim_his = np.zeros((max_epochs, 1))
val_ssim_his = np.zeros((max_epochs, 1))
for epoch in range(epoch_start, max_epochs):
    train_loss = utils.Metric('train_loss')
    val_loss = utils.Metric('val_loss')
    train_ssim = utils.Metric('train_ssim')
    val_ssim = utils.Metric('val_ssim')
    # Training
    with tqdm(total=len(train_loader),
            desc='Epoch  #{}'.format(epoch + 1),
            disable=not verbose) as t:
        for batch_idx, data in enumerate(train_loader):

            # utils.adjust_learning_rate_3_zones(epoch, 40, 80, 120, args, optimizer, hvd.size())
            utils.adjust_learning_rate(optimizer, epoch, args, hvd.size(), 0.1, 20)

            if use_cuda: data = data.cuda()

            data = data.repeat(1, n_coil, 1, 1)  # (data, n_coil, axis=1)  # shaped (320, 8, 260, 311)
            # Transfer to gpu device and into k-space measurement with noise.
            data = data * sensi_map
            data = torch.fft.fftshift(input=torch.fft.fft2(data), dim=(2, 3))

            data_k_full = torch.normal(data, n_std)
            if acceleration > 1:
                data_k = data_k_full * mask  # local batch has shape of ([320, 8, 260, 311]) and is complex array
            data_recs = torch.fft.ifft2(data_k)
            data_cat = torch.square(torch.abs(data_recs))
            data_cat = torch.sqrt(torch.sum(input=data_cat, dim=1))
            data_cat = torch.reshape(data_cat, (allreduce_batch_size, 1, dim1, dim2))

            target_recs = torch.fft.ifft2(data_k_full)
            target_cat = torch.square(torch.abs(target_recs))
            target_cat = torch.sqrt(torch.sum(input=target_cat, dim=1))
            target_cat = torch.reshape(target_cat, (allreduce_batch_size, 1, dim1, dim2))

            if args.model_name == 'unet':
                data_cat = F.pad(data_cat, pad)
                target_cat = F.pad(target_cat, pad)

            model.train()
            optimizer.zero_grad()
            output = model(data_cat)
            loss = mse_loss(output, target_cat)
            loss.backward()
            optimizer.step()

            ssim_value = pytorch_msssim.ssim(output, target_cat, data_range=1.4, size_average=True, nonnegative_ssim=True) # only for monitoring purpose
            train_loss.update(loss, hvd)
            train_ssim.update(ssim_value, hvd)

            if batch_idx + 1 == len(train_loader):
                model.eval()
                with torch.no_grad():
                    for data in val_loader:
                        if use_cuda:
                            data = data.cuda()

                        data = data.repeat(1, n_coil, 1, 1)  # (data, n_coil, axis=1)  # shaped (320, 8, 260, 311)
                        # Transfer to gpu device and into k-space measurement with noise.
                        data = data * sensi_map
                        data = torch.fft.fftshift(input=torch.fft.fft2(data), dim=(2, 3))

                        data_k = torch.normal(data, n_std)
                        if acceleration > 1:
                            data_k = data_k * mask  # local batch has shape of ([320, 8, 260, 311]) and is complex array
                        data_recs = torch.fft.ifft2(data_k)
                        data_cat = torch.square(torch.abs(data_recs))
                        data_cat = torch.sqrt(torch.sum(input=data_cat, dim=1))
                        data_cat = torch.reshape(data_cat, (-1, 1, dim1, dim2))

                        target_k = torch.normal(data, n_std)
                        target_recs = torch.fft.ifft2(target_k)
                        target_cat = torch.square(torch.abs(target_recs))
                        target_cat = torch.sqrt(torch.sum(input=target_cat, dim=1))
                        target_cat = torch.reshape(target_cat, (-1, 1, dim1, dim2))

                        if args.model_name == 'unet':
                            data_cat = F.pad(data_cat, pad)
                            target_cat = F.pad(target_cat, pad)

                        output = model(data_cat)
                        loss = mse_loss(output, target_cat)

                        ssim_value = pytorch_msssim.ssim(output, target_cat, data_range=1.4, size_average=True, nonnegative_ssim=True)
                        val_loss.update(loss, hvd)
                        val_ssim.update(ssim_value, hvd)
                t.set_postfix({'train loss': train_loss.avg.item(), \
                               'val loss': val_loss.avg.item(), \
                               'train ssim': train_ssim.avg.item(), \
                               'val ssim': val_ssim.avg.item()})
                t.update(len(train_loader))

    if (args.save_log_ckpts):
        utils.save_checkpoint(epoch, args, hvd, model, optimizer)
        train_loss_his[epoch] = train_loss.avg.item()
        val_loss_his[epoch] = val_loss.avg.item()
        train_ssim_his[epoch] = train_ssim.avg.item()
        val_ssim_his[epoch] = val_ssim.avg.item()
        utils.save_log_to_h5(args, hvd, \
                             train_loss_his[train_loss_his != 0], \
                             val_loss_his[val_loss_his != 0], \
                             train_ssim_his[train_ssim_his != 0], \
                             val_ssim_his[val_ssim_his != 0])
