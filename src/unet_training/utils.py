
import os
import numpy as np
import torch
import pickle
import h5py
import torch.nn as nn
import torch.nn.functional as F

# wd: 20, 40
def adjust_learning_rate_3_zones(epoch, ep1, ep2, ep3, args, optimizer, nGPUS):
# Adjusts the learning rate based on the current epoch using a 3-zone schedule.
#
# Args:
#     epoch (int): Current epoch number.
#     ep1, ep2, ep3 (int): Epoch thresholds for learning rate changes.
#     args: Argument object containing base_lr and batches_per_allreduce.
#     optimizer (torch.optim.Optimizer): The optimizer to update.
#     nGPUS (int): Number of GPUs used in training.
#
# Returns:
#     None. The function modifies the optimizer's learning rate in-place
    if epoch < ep1:
        lr_adj = 1
    elif epoch < ep2:
        lr_adj = 1e-1
    elif epoch < ep3:
        lr_adj = 1e-2
    else:
        lr_adj = 1e-3
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.base_lr * nGPUS * args.batches_per_allreduce * lr_adj

def adjust_learning_rate(opt, epo, args, nGPUs, factor, epo_threshold):
        """Sets the learning rate to the initial LR decayed by $factor$ every $epo_threshold$ epochs"""
        lr_adj = factor ** (epo // epo_threshold)
        for param_group in opt.param_groups:
            param_group['lr'] = args.base_lr * nGPUs * args.batches_per_allreduce * lr_adj

# Horovod: average metrics from distributed training.
class Metric(object):
# A class for tracking and averaging metrics in distributed training.
#
# Attributes:
#     name (str): The name of the metric.
#     sum (torch.Tensor): The sum of all metric values.
#     n (torch.Tensor): The number of metric values added.
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val, hvd):
        # here values such as mertic loss from outside training/validation
        # loop for each batch is extracted.
        self.sum += hvd.allreduce(val.detach().cpu(), name=self.name)
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n


# save checkpoint from rank 0 so that weights from other ranks are
# not repeated
def save_checkpoint(epoch, args, hvd, model, optimizer):
    if hvd.rank() == 0:
        filepath = args.checkpoint_format.format(epoch=epoch + 1)
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, filepath)


# save log from rank 0
def save_log(epoch, args, hvd, loss_acc):
    if hvd.rank() == 0:
        filepath = args.log_file_format.format(epoch=epoch + 1)
        with open(filepath, 'wb') as f:
            pickle.dump(loss_acc, f)


# save log to a single h5 file from rank 0
def save_log_to_h5(args, hvd, train_loss, val_loss, train_ssim, val_ssim):
    if hvd.rank() == 0:
        filepath = args.log_file_format
        f = h5py.File(filepath, "w")
        f.create_dataset('train_loss', data=train_loss)
        f.create_dataset('val_loss', data=val_loss)
        f.create_dataset('train_ssim', data=train_ssim)
        f.create_dataset('val_ssim', data=val_ssim)

# save log to a single h5 file from rank 0
def save_log_to_h5_loss(args, hvd, train_loss, val_loss):
    if hvd.rank() == 0:
        filepath = args.log_file_format
        f = h5py.File(filepath, "w")
        f.create_dataset('train_loss', data=train_loss)
        f.create_dataset('val_loss', data=val_loss)


def print_model_summary(model, input_size):
# Print a summary of the model architecture, including layer shapes and parameter counts.
#
# Args:
#     model (nn.Module): The PyTorch model to summarize.
#     input_size (tuple): The input size for the model (e.g., (1, 28, 28) for MNIST).
#
# Example usage
# print_model_summary(model, (1, 28, 28))
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)
            m_key = f'{class_name}-{module_idx+1}'
            summary[m_key] = {
                "input_shape": list(input[0].size()),
                "output_shape": list(output.size()),
                "nb_params": sum(p.numel() for p in module.parameters())
            }
        if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList) and module != model:
            hooks.append(module.register_forward_hook(hook))

    summary = {}
    hooks = []
    model.apply(register_hook)
    with torch.no_grad():
        model(torch.zeros(1, *input_size))

    for h in hooks:
        h.remove()

    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    print(line_new)
    print("================================================================")
    total_params = 0
    for layer in summary:
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"])
        )
        total_params += summary[layer]["nb_params"]
        print(line_new)
    print("================================================================")
    print(f"Total params: {total_params:,}")
    print("----------------------------------------------------------------")


