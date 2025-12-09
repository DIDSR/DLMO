# Predict denoised images using a trained CNN denoiser
# Command-line Options:
#     --task: Task type (detection/rayleigh). Default is 'rayleigh'.
#     --test-path: Path to noisy images for testing.
#     --acceleration: Acceleration factor (2, 4, 6, or 8).
#     --model_name: CNN denoiser model (cnn3, redcnn, udncnn, dncnn, unet).
#     --num-channels: Number of channels (1 for grayscale, 3 for RGB). Default is 1.
#     --batch-size: Batch size for testing.
#     --batches-per-allreduce: Number of batches processed locally before allreduce. Default is 1.
#     --fp16-allreduce: Use fp16 compression during allreduce (flag).
#     --pretrained-model-path: Path to the directory containing the pre-trained model.
#     --pretrained-model-checkpoint-format: Format of the checkpoint file. Default is 'checkpoint-{epoch}.pth.tar'.
#     --pretrained-model-epoch: Epoch number of the pre-trained model to use. Default is 150.
#
# Note: Ensure that the specified paths, model names, and other parameters match your setup and available resources.
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
import models

import os
import math
import sys

from dataset import DatasetFromHdf5
import scipy.io
import utils
from tqdm import tqdm
import add_signals

# from torchsummary import summary

import time

start_time = time.time()
# --------------------------------------Some basic settings ---------------------------------------#
parser = argparse.ArgumentParser(description='Predict denoised images using a trained CNN denoiser.')
parser.add_argument('--task', default='rayleigh', help='Task type (detection/rayleigh).')
parser.add_argument('--test-path', \
                    help='Path to noisy images for test.')
parser.add_argument('--acceleration', type=int, default=2, \
                    help='Acceleration factor in range of 2 to 12.')
parser.add_argument('--model_name', help='CNN denoiser model.')
parser.add_argument('--num-channels', type=int, default=1, help='3 for rgb images and 1 for gray scale images')
parser.add_argument('--batch-size', help='Batch size.', type=int)
parser.add_argument('--batches-per-allreduce', type=int, default=1,
                    help='number of batches processed locally before executing allreduce across workers;'
                    'It multiplies the total batch size. (RHR: 1 loss function eqs 1 batches-per-allreduce')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--pretrained-model-path', help='The previous trained model (provide path).')
parser.add_argument('--pretrained-model-checkpoint-format', default='checkpoint-{epoch}.pth.tar',
                    help='checkpoint file format')
parser.add_argument('--pretrained-model-epoch', type=int, default=150, help='Transfered learning based on a previous trained model (provide epoch).')

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

# ----------------------------Training settings------------------------------------------------------------#
model_name = args.model_name
num_channels = args.num_channels

batch_size = args.batch_size
batches_per_allreduce = args.batches_per_allreduce
allreduce_batch_size = batch_size * batches_per_allreduce

test_data_path = args.test_path

output_path = "./ai_rec_prediction/" + model_name + "_" + task_type + "_acc_" + str(acceleration) + "/"
if not os.path.isdir(output_path): os.makedirs(output_path, exist_ok=True)

dim1, dim2 = 260, 311
pad = (4, 5, 6, 6) # zero-padding for unet training
cmpr_dtype = 'float32'

test_half_size = 4000
test_tot_size = 2 * test_half_size

if cmpr_dtype == 'float16':
    torch_dtype = torch.float16
else:
    torch_dtype = torch.float32

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
# Load Model and objective loss function
# ==================================================================
if hvd.rank() == 0: print("Load model .... \n")

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
    model = UNet()
else:
    print("ERROR! Re-check DNN model (architecture) string!")
    sys.exit()

# transfer models to cuda
if use_cuda:
    model = model.cuda()

criterion = nn.MSELoss()

# Restore from a trained model.
# Horovod: restore on the first worker which will broadcast weights to other workers.
pretrained_model_path = args.pretrained_model_path + '/'
pretrained_model_epoch = args.pretrained_model_epoch
pretrained_model_checkpoint_format = args.pretrained_model_checkpoint_format.format(epoch=pretrained_model_epoch)
pretrained_model = os.path.join(pretrained_model_path, pretrained_model_checkpoint_format)
if hvd.rank() == 0:
    checkpoint = torch.load(pretrained_model)
    model.load_state_dict(checkpoint['model'])
    print("Loaded trained network from:", pretrained_model, flush=True)

# Horovod: broadcast parameters & optimizer state.
hvd.broadcast_parameters(model.state_dict(), root_rank=0)

# ==================================================================
# load test data
# ==================================================================
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

test_data_file = test_data_path + "/" + \
                 "test_acc" + str(acceleration) + "_rsos.hdf5"
if hvd.rank() == 0: print("\nReading the test dataset from: " + test_data_file, flush=True)

test_dataset = DatasetFromHdf5(hvd=hvd, file_path=test_data_file, \
                              mod_num=hvd.size() * batch_size)
test_sampler = torch.utils.data.distributed.DistributedSampler(
    test_dataset, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                         sampler=test_sampler, **kwargs)

if hvd.rank() == 0:
    print('%d denoisy images are used to test CNN denoiser\n' % (len(test_dataset.data)), flush=True)  #
    print('min and max in test data: [%.4f, %.4f]' % (np.min(test_dataset.data), np.max(test_dataset.data)), flush=True)
    print('Shape of the loaded test data is:', test_dataset.data[0].shape, 'and its dtype is:', test_dataset.data[0].dtype, flush=True)

# ==================================================================
# test the model
# ==================================================================
if hvd.rank() == 0: print("Denoising images .... ")
test_loss = utils.Metric('test_loss')

noisy_imgs = np.empty([test_tot_size, dim1, dim2])
preds = np.empty([test_tot_size, dim1, dim2])
# i  = 0
with tqdm(total=len(test_loader),
        disable=not verbose) as t:
    for batch_idx, data in enumerate(test_loader):

        if use_cuda: data = data.cuda()

        model.eval()

        if args.model_name == 'unet':
            data = F.pad(data, pad)

        output = model(data)
        if args.model_name == 'unet':
            output = output[:,:, 6:6 + dim1, 4:dim2 + 4]
            data = data[:,:, 6:6 + dim1, 4:dim2 + 4]
        preds[batch_size * batch_idx:batch_size * (batch_idx + 1),:,:] = np.squeeze(output.cpu().detach().numpy())
        noisy_imgs[batch_size * batch_idx:batch_size * (batch_idx + 1),:,:] = np.squeeze(data.cpu().detach().numpy())

        # i = i + 1
        t.update(1)

np.save(output_path + "/preds_detection.npy", preds)
np.save(output_path + "/noisy_detection.npy", noisy_imgs)

# save a few examples
# np.save(output_path + "/preds_examples_0_L" + str(L) + ".npy", preds[:10,:,:])
# np.save(output_path + "/noisy_examples_0_L" + str(L) + ".npy", noisy_imgs[:10,:,:])
# np.save(output_path + "/preds_examples_1_L" + str(L) + ".npy", preds[int(len(test_dataset.data) / 2):int(len(test_dataset.data) / 2) + 10,:,:])
# np.save(output_path + "/noisy_examples_1_L" + str(L) + ".npy", noisy_imgs[int(len(test_dataset.data) / 2):int(len(test_dataset.data) / 2) + 10,:,:])
