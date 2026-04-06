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
from models import Net_7conv2_dropout as DLMO_Net
from sklearn.metrics import roc_auc_score

import os
import math
import sys

from dataset import DatasetFromHdf5
import scipy.io
import utils
from tqdm import tqdm

from torchsummary import summary

import time

start_time = time.time()
# --------------------------------------Some basic settings -----------------------------------------------------#
parser = argparse.ArgumentParser(description='Applying a trained DLMO for detection/descrimination task')
parser.add_argument('--task', default='rayleigh', help='Task type (detection/rayleigh).')
parser.add_argument('--test-path', help='Path to reconstructed MR images with signals.')
parser.add_argument('--cnn-denoiser-name', \
                    help='Name of cnn denoiser that denoised the MR images.')
parser.add_argument('--acceleration', type=int, default=4, \
                    help='Acceleration factor in range of 2 to 12.')
parser.add_argument('--num-channels', type=int, default=1, help='3 for rgb images and 1 for gray scale images')
parser.add_argument('--batch-size', help='Batch size.', type=int)
parser.add_argument('--pretrained-model-path', help='The previous trained path to DLMO weights.')
parser.add_argument('--pretrained-model-checkpoint-format', default='checkpoint-{epoch}.pth.tar',
                    help='checkpoint file format')
parser.add_argument('--pretrained-model-epoch', type=int, default=150, help='Epoch number of the pretrained \
                    DLMO checkpoint file.')
parser.add_argument('--out-tag', default=None, help='Optional tag for the output filename (saved as preds_<out_tag>.npy).')
args = parser.parse_args()

task_type = args.task  # task type (detection/rayleigh)
if task_type not in ['detection', 'rayleigh']:
    print('Invalid task type.', flush=True)
    sys.exit()

acceleration = args.acceleration  # acceleration factor
if acceleration not in [1, 4, 8]:
    print('Warning! For acceleration factors other than 4, 8 train your \
        DL postprocessor accordingly and save the checkpoints in the \
        trained_model folder.')

# ----------------------------CMD Line Inputs-----------------------------------------------------#
num_channels           = args.num_channels
batch_size             = args.batch_size
test_data_in_path      = args.test_path
output_path            = "./dlmo_discrimination/acc" + str(acceleration)
if not os.path.isdir(output_path): os.makedirs(output_path, exist_ok=True)

#input file path with MR reconstruction ---------------------------------------------------------#
if args.cnn_denoiser_name is None:
    if acceleration == 1:
         test_data_in_file    = test_data_in_path + "/test_acc4_at_acc" + str(acceleration) + "_rsos.hdf5"   # rsos (1x) recon
    else:
        test_data_in_file     = test_data_in_path + "/test_acc" + str(acceleration) + "_rsos.hdf5"           # rsos (4x) recon
else:
    test_data_in_file         = (test_data_in_path + "/test_acc" + str(acceleration) + "_" +         
                                args.cnn_denoiser_name +".hdf5")                                              # unet recon
# input dimension and dtype info --------
dim1, dim2     = 260, 311
cmpr_dtype     = 'float32'

if cmpr_dtype == 'float16':
    torch_dtype = torch.float16
else:
    torch_dtype = torch.float32

# retrieving the checkpoint path -------------------------------------------------------------------------------------
pretrained_model_checkpoint_format = args.pretrained_model_checkpoint_format.format(epoch=args.pretrained_model_epoch)
pretrained_model_path              = os.path.join(args.pretrained_model_path, pretrained_model_checkpoint_format)
hvd.init()

# --------------------- CUDA for PyTorch --------------------------------#
use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

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
  print('pretrained model full path:', pretrained_model_path)
  print("testing data full path    : ", test_data_in_file)
  print('\n----------------------------------------\n')

# Horovod: print logs on the first worker
verbose = 0 if hvd.rank() == 0 else 0

# ==================================================================
# Load Model and objective loss function
# ==================================================================
if hvd.rank() == 0: print("Load model .... \n")

model = DLMO_Net(dim1=dim1, dim2=dim2)

# transfer models to cuda
if use_cuda: model = model.cuda()

if hvd.rank() == 0:
    checkpoint = torch.load(pretrained_model_path)
    model.load_state_dict(checkpoint['model'])
    print("Loaded trained network from:", pretrained_model_path, flush=True)

# Horovod: broadcast parameters & optimizer state.
hvd.broadcast_parameters(model.state_dict(), root_rank=0)

# ==================================================================
# load test data
# ==================================================================
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
test_dataset = DatasetFromHdf5(hvd=hvd, file_path=test_data_in_file, \
                              mod_num=hvd.size() * batch_size)
test_sampler = torch.utils.data.distributed.DistributedSampler(
    test_dataset, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                         sampler=test_sampler, **kwargs)

if hvd.rank() == 0:
    print('Trained DLMO is applied to %d MR images\n' % (len(test_dataset.data)), flush=True)  
    print('Data range [min, max] in this set: [%.4f, %.4f]' % (np.min(test_dataset.data), np.max(test_dataset.data)), flush=True)
    print('Shape of the loaded test data is:', test_dataset.data.shape, 'and its dtype is:', test_dataset.data.dtype, flush=True)
    print('Below is the DLMO architecture')
    summary(model, (1, dim1, dim2))
    test_tot_size   = test_dataset.data.shape[0]
    test_half_size  = int(test_tot_size/2)

# ==================================================================
# test the model
# ==================================================================
if hvd.rank() == 0: print("Evaluating DLMO on the Rayleigh task .... ")

preds = np.empty(0)
with tqdm(total=len(test_loader),
        disable=not verbose) as t:
    for batch_idx, (data, target) in enumerate(test_loader):
        if use_cuda: data = data.cuda()
        model.eval()
        output = model(data)
        preds  = np.append(preds, np.squeeze(output.cpu().detach().numpy()), axis=0)
        t.update(1)

# ------------------------------------- Results ------------------------------------------#
if hvd.rank() == 0:
    if args.cnn_denoiser_name is None:
        if args.out_tag is not None:
            pred_fname = "preds_" + args.out_tag + ".npy"
        else:
            pred_fname = "preds_rsos.npy"
        if hvd.rank() == 0: print("Saving outputs to: " + output_path + "/" + pred_fname)
        np.save(output_path + "/" + pred_fname, preds)
    else:
        if args.out_tag is not None:
            pred_fname = "preds_" + args.out_tag + ".npy"
        else:
            pred_fname = "preds_" + args.cnn_denoiser_name + ".npy"
        print("Saving outputs to: " + output_path + "/" + pred_fname)
        np.save(output_path + "/" + pred_fname, preds)

    # pred array stores H_d pred and then H_s pred --------------------------------------------------------
    auc = roc_auc_score(np.concatenate((np.zeros(test_half_size), np.ones(test_half_size)), axis=0), preds)
    print("Acceleration factor: " + str(acceleration) + " AUC: " + str(auc))
