# Train the deep learning-based model observer.
# This version removes the Horovod dependency and runs as a standard
# single-process PyTorch training script.

import argparse
import os
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import DatasetFromHdf5
from models import Net_7conv2_dropout as Net
import utils


class _DummyCompression:
    fp16 = "fp16"
    none = "none"


class DummyHvd:
    """Minimal Horovod-like shim for single-process execution."""

    Compression = _DummyCompression

    def init(self):
        return None

    def rank(self):
        return 0

    def size(self):
        return 1

    def local_rank(self):
        return 0

    def allreduce(self, tensor, name=None):
        return tensor

    def DistributedOptimizer(self, optimizer, named_parameters=None, compression=None,
                             backward_passes_per_step=1):
        return optimizer

    def broadcast_parameters(self, state_dict, root_rank=0):
        return None

    def broadcast_optimizer_state(self, optimizer, root_rank=0):
        return None


hvd = DummyHvd()


# --------------------------------------Some basic settings ---------------------------------------#
parser = argparse.ArgumentParser(description='Train the DLMO.')
parser.add_argument('--acceleration', help='Acceleration factor ([1,2,4,6,8,10,12]).', type=int)
parser.add_argument('--train-data-path', help='Path to reconstructed MR images (with signals) for training DLMO.', type=str)
parser.add_argument('--val-data-path', help='Path to few samples of reconstructed MR images (with signals) for tuning DLMO.')
parser.add_argument('--output-path', type=str, default='trained_model', help='Path to save checkpoints and log files')
parser.add_argument('--pretrained-model-path', help='Transfer learning based on a previous trained model (provide path).')
parser.add_argument('--pretrained-model-checkpoint-format', default='checkpoint-{epoch}.pth.tar', help='checkpoint file format')
parser.add_argument('--pretrained-model-epoch', type=int, default=170, help='Epoch for the pretrained model checkpoint.')
parser.add_argument('--nepochs', type=int, default=50, help='number of epochs to train')
parser.add_argument('--batch-size', help='Training batch size.', type=int, required=True)
parser.add_argument('--val-batch-size', type=int, default=16, help='batch size for validation/tuning data.')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay a.k.a regularization on weights')
parser.add_argument('--shuffle_patches', action='store_true', help='Shuffles train/validation patch pairs at DataLoader level.')
parser.add_argument('--fp16-allreduce', action='store_true', default=False, help='Retained for CLI compatibility; ignored without Horovod.')
parser.add_argument('--checkpoint-format', default='checkpoint-{epoch}.pth.tar', help='checkpoint file format')
parser.add_argument('--save-log-ckpts', action='store_true', help='saves log writer and checkpoints')
parser.add_argument('--log-file-format', default='log.h5', help='log file format')

args = parser.parse_args()

acceleration = args.acceleration
if acceleration not in [1, 2, 4, 6, 8]:
    print('Invalid acceleration factor.', flush=True)
    sys.exit(1)

# CMD variables ---------------------------------------------
batch_size = args.batch_size
val_batch_size = args.val_batch_size
batches_per_allreduce = 1
allreduce_batch_size = batch_size * batches_per_allreduce
max_epochs = args.nepochs
epoch_start = 0

# Paths to i/o files --------------------------------------
train_data_path = args.train_data_path
val_data_path = args.val_data_path
args.output_path = args.output_path + str(acceleration) + '_single/'
output_path = args.output_path
pretrained_model_path = args.pretrained_model_path
pretrained_model_epoch = args.pretrained_model_epoch

cmpr_dtype = 'float32'
torch_dtype = torch.float16 if cmpr_dtype == 'float16' else torch.float32

if args.save_log_ckpts:
    os.makedirs(output_path, exist_ok=True)
    checkpoint_folder = os.path.join(output_path, 'checkpoints')
    os.makedirs(checkpoint_folder, exist_ok=True)
    args.checkpoint_format = os.path.join(checkpoint_folder, args.checkpoint_format)

    current_log_folder = os.path.join(output_path, 'logs')
    os.makedirs(current_log_folder, exist_ok=True)
    if '{epoch}' in args.log_file_format:
        args.log_file_format = os.path.join(current_log_folder, args.log_file_format.format(epoch='final'))
    else:
        args.log_file_format = os.path.join(current_log_folder, args.log_file_format)

hvd.init()

# ------------------------------------- CUDA for PyTorch ------------------------------------------#
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
if use_cuda:
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.set_device(hvd.local_rank())
cudnn.benchmark = use_cuda

# Display command line arguments
if hvd.rank() == 0:
    print('\n----------------------------------------')
    print('Command line arguments')
    print('----------------------------------------')
    print('\nNo. of gpus visible to this process:', torch.cuda.device_count() if use_cuda else 0)
    print('Training world size:', hvd.size())
    for key, value in vars(args).items():
        print(f'{key}: {value}')
    print('\n----------------------------------------\n')

verbose = 1 if hvd.rank() == 0 else 0
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# ==================================================================
# load training data
# ==================================================================
train_dataset = DatasetFromHdf5(hvd=None, file_path=train_data_path, mod_num=batch_size)
train_loader = DataLoader(
    train_dataset,
    batch_size=allreduce_batch_size,
    shuffle=args.shuffle_patches,
    drop_last=False,
    **kwargs,
)

dim1 = train_dataset.data.shape[2]
dim2 = train_dataset.data.shape[3]

if hvd.rank() == 0:
    print(' Dimension of training (input <-> target) batches is: {} <-> {}'.format(
        train_dataset.data.shape, train_dataset.target.shape
    ))

# ==================================================================
# load validation data
# ==================================================================
val_dataset = DatasetFromHdf5(hvd=None, file_path=val_data_path, mod_num=val_batch_size)
val_loader = DataLoader(
    val_dataset,
    batch_size=val_batch_size,
    shuffle=args.shuffle_patches,
    drop_last=False,
    **kwargs,
)

if hvd.rank() == 0:
    print(' Dimension of validation (input <-> target) batches is: {} <-> {}'.format(
        val_dataset.data.shape, val_dataset.target.shape
    ))

# ==================================================================
# Initializing Model and objective loss function
# ==================================================================
if hvd.rank() == 0:
    print('Building model .... ')

model = Net(dim1=dim1, dim2=dim2, filter_size=7).to(device=device, dtype=torch_dtype)

optimizer = optim.Adam(
    model.parameters(),
    lr=(1e-5 * batches_per_allreduce * hvd.size()),
    weight_decay=args.wd,
)

criterion = nn.BCEWithLogitsLoss()

# Restore from a previous checkpoint, if initial_epoch is specified.
if pretrained_model_path:
    filepath = os.path.join(
        pretrained_model_path,
        args.pretrained_model_checkpoint_format.format(epoch=pretrained_model_epoch),
    )
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model'])
    optimizer_state = checkpoint.get('optimizer')
    if optimizer_state is not None and acceleration == 1:
        optimizer.load_state_dict(optimizer_state)
    print('Loaded pretrained network from:', filepath, flush=True)

# ==================================================================
# train the model
# ==================================================================
if hvd.rank() == 0:
    print('Training .... ')

train_acc_arr = np.zeros((max_epochs, 1), dtype=np.float32)
val_acc_arr = np.zeros((max_epochs, 1), dtype=np.float32)
train_loss_arr = np.zeros((max_epochs, 1), dtype=np.float32)
val_loss_arr = np.zeros((max_epochs, 1), dtype=np.float32)


def _safe_roc_auc(target_tensor, output_tensor):
    y_true = np.squeeze(target_tensor.detach().cpu().numpy())
    y_score = np.squeeze(output_tensor.detach().cpu().numpy())
    y_true = np.atleast_1d(y_true)
    y_score = np.atleast_1d(y_score)
    if np.unique(y_true).size < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_score))


def train(epoch):
    model.train()
    train_loss = utils.Metric('train_loss')
    train_acc = utils.Metric('train_acc')
    with tqdm(total=len(train_loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', desc='Epoch  #{}'.format(epoch + 1), disable=not verbose) as t:
        # this is automatically performed in a batch-wise manner
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device=device, dtype=torch_dtype)
            target = target.to(device=device, dtype=torch.float32)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss.update(loss, hvd)
            roc_acc = torch.tensor(_safe_roc_auc(target, output), dtype=torch.float32)
            train_acc.update(roc_acc, hvd)
            #print('batch_idx:',batch_idx, 'train acc:', train_acc.avg.item(), 'train loss:', train_loss.avg.item())
            t.set_postfix({'train acc': train_acc.avg.item(), 'train loss': train_loss.avg.item()})
            t.update(1)
    if args.save_log_ckpts:
        train_acc_arr[epoch] = train_acc.avg.item()
        train_loss_arr[epoch] = train_loss.avg.item()


@torch.no_grad()
def validate(epoch):
    model.eval()
    val_loss = utils.Metric('val_loss')
    val_acc = utils.Metric('val_acc')
    with tqdm(total=len(val_loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', desc='Val    #{}'.format(epoch + 1), disable=not verbose) as t:
        for batch_idx, (data, target) in enumerate(val_loader):
            data = data.to(device=device, dtype=torch_dtype)
            target = target.to(device=device, dtype=torch.float32)

            output = model(data)
            loss = criterion(output, target)
            val_loss.update(loss, hvd)

            roc_acc = torch.tensor(_safe_roc_auc(target, output), dtype=torch.float32)
            val_acc.update(roc_acc, hvd)
            #print('batch_idx:',batch_idx, 'val acc:', val_acc.avg.item(), 'val loss:', val_loss.avg.item())
            t.set_postfix({'val acc': val_acc.avg.item(), 'val loss': val_loss.avg.item()})
            t.update(1)

    if args.save_log_ckpts:
        val_acc_arr[epoch] = val_acc.avg.item()
        val_loss_arr[epoch] = val_loss.avg.item()


# ====================================================================
# Main function with train, validate; save weights and metrics
# ======================================================================
for epoch in range(epoch_start, max_epochs):
    train(epoch)
    validate(epoch)

    if args.save_log_ckpts:
        utils.save_checkpoint(epoch, args, hvd.rank(), model, optimizer)

if args.save_log_ckpts:
    loss_acc = {
        'train acc': train_acc_arr,
        'val acc': val_acc_arr,
        'train loss': train_loss_arr,
        'val loss': val_loss_arr,
    }
    utils.save_log_to_h5(args, hvd.rank(), loss_acc)
