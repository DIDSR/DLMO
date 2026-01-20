# Estimate the probability of doublet signal using a trained deep learning-based model observer.
# It supports Rayleigh discrimination tasks, and can handle both regular and CNN-denoised images.
#
# The script uses Horovod for distributed training and PyTorch for the neural network implementation.
#
# Main components:
# 1. Argument parsing
# 2. Model loading and initialization
# 3. Data loading
# 4. Model evaluation
# 5. Results saving and AUC calculation
#
# To use, source the following environment.
# source /anaconda3/base_env.sh
# source /anaconda3/horovod_sm80_env.sh

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
from sklearn.metrics import roc_auc_score

import os
import math
import sys

from dataset import DatasetFromHdf5, DatasetFromNpz
import scipy.io
import utils
from tqdm import tqdm

# from torchsummary import summary

import time

start_time = time.time()
# --------------------------------------Some basic settings ---------------------------------------#
parser = argparse.ArgumentParser(description='Estimate the probability of signal existing using a trained CNN IO.')
parser.add_argument('--task', default='rayleigh', help='Task type (detection/rayleigh).')
parser.add_argument('--test-path', \
                    help='Path to images for test.')
parser.add_argument('--is-cnn-denoised', \
                    action='store_true', default=False,
                    help='Whether the input images are cnn-denoised.')
parser.add_argument('--test-cnn-denoiser', \
                    help='Name of cnn denoiser that denoised images.')
parser.add_argument('--acceleration', type=int, default=2, \
                    help='Acceleration factor in range of 2 to 12.')
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
if acceleration not in [1, 2, 4, 6, 8]:
    print('Invalid acceleration factor.', flush=True)
    sys.exit()

num_channels = args.num_channels

batch_size = args.batch_size
batches_per_allreduce = args.batches_per_allreduce
allreduce_batch_size = batch_size * batches_per_allreduce

test_data_path = args.test_path
cnn_model_name = args.test_cnn_denoiser

output_path = "./dlmo_predictions/acc" + str(acceleration) + "/"
if not os.path.isdir(output_path): os.makedirs(output_path, exist_ok=True)

dim1, dim2 = 260, 311
cmpr_dtype = 'float32'

test_half_size = 3
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
verbose = 0 if hvd.rank() == 0 else 0

# ==================================================================
# Load Model and objective loss function
# ==================================================================
if hvd.rank() == 0: print("Load model .... \n")

model = Net(dim1=dim1, dim2=dim2)

# transfer models to cuda
if use_cuda:
    model = model.cuda()

# Restore from a trained model.
# Horovod: restore on the first worker which will broadcast weights to other workers.
pretrained_model_path = args.pretrained_model_path
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

if not args.is_cnn_denoised:
    test_data_file = test_data_path + "/" + \
                     "accelerated_acc" + str(acceleration) + "_rsos.hdf5"
    if hvd.rank() == 0: print("\nReading the test dataset from: " + test_data_file, flush=True)

    test_dataset = DatasetFromHdf5(hvd=hvd, file_path=test_data_file, \
                                   mod_num=hvd.size() * batch_size)
    test_dataset = test_dataset.data
else:
    test_data_file = test_data_path + "/" + \
                     cnn_model_name + "_" + task_type + "_acc_" + str(acceleration) + "/" + \
                     "preds.npy"
    if hvd.rank() == 0: print("\nReading the test dataset from: " + test_data_file, flush=True)

    test_dataset = DatasetFromNpz(test_data_file)
    test_dataset = test_dataset.data
    test_dataset = np.reshape(test_dataset, (test_dataset.shape[0], 1, dim1, dim2)).astype('float32')

test_sampler = torch.utils.data.distributed.DistributedSampler(
    test_dataset, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                         sampler=test_sampler, **kwargs)

if hvd.rank() == 0:
    print('%d images are used to test CNN IO\n' % (len(test_dataset)), flush=True)  #
    print('min and max in test data: [%.4f, %.4f]' % (np.min(test_dataset), np.max(test_dataset)), flush=True)
    print('Shape of the loaded test data is:', test_dataset.shape, 'and its dtype is:', test_dataset.dtype, flush=True)

# ==================================================================
# test the model
# ==================================================================
if hvd.rank() == 0: print("Evaluating on the Rayleigh task .... ")

preds = np.empty(0)
with tqdm(total=len(test_loader),
        disable=not verbose) as t:
    for batch_idx, data in enumerate(test_loader):

        if use_cuda: data = data.cuda()

        model.eval()

        output = model(data)
        preds = np.append(preds, np.squeeze(output.cpu().detach().numpy()), axis=0)

        t.update(1)

# ------------------------------------- Results ------------------------------------------#

if not args.is_cnn_denoised:
    if hvd.rank() == 0: print("Saving outputs to: " + output_path + "/preds_rsos.npy")
    np.save(output_path + "/preds_rsos.npy", preds)
else:
    if hvd.rank() == 0: print("Saving outputs to: " + output_path + "/preds_" + cnn_model_name + ".npy")
    np.save(output_path + "/preds_" + cnn_model_name + ".npy", preds)

auc = roc_auc_score(np.concatenate((np.ones(test_half_size), np.zeros(test_half_size)), axis=0), preds)
if hvd.rank() == 0: print("Acceleration factor: " + str(acceleration) + " AUC: " + str(auc))
