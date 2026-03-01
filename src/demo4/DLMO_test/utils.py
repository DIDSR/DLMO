
import os
import numpy as np
import torch
import pickle
import h5py

def list_all_npy_files(super_path, cmpr_dtype=None, unity_normalize=False):
# Recursively reads all .npz files in a directory structure and combines them into a single numpy array.
#
# Args:
# super_path (str): The root directory to start searching for .npz files.
# cmpr_dtype (numpy dtype, optional): If provided, converts the arrays to this data type.
# unity_normalize (bool): If True, normalizes uint8 data to the range [0, 1].
#
# Returns:
# numpy.ndarray: A combined array of all .npz files found.

    arr_accumulator = []
    idx = 0
    for main_folder in os.listdir(super_path):  # outermost folder
        for sub_folder in os.listdir(os.path.join(super_path, main_folder)):  # sub folders
            inner_path = os.path.join(super_path, main_folder, sub_folder)  # each inner folder
            for file in os.listdir(inner_path):
                if file.endswith('.npz'):
                    each_npz_file = os.path.join(inner_path, file)
                    each_npz_read = np.load(each_npz_file)
                    each_npz_arr = each_npz_read.f.arr_0
                    each_npz_arr = np.squeeze(each_npz_arr)

                    # change the datatype and normalize the 'uint8'
                    if cmpr_dtype != None: each_npz_arr = each_npz_arr.astype(cmpr_dtype)
                    if unity_normalize: each_npz_arr = each_npz_arr / 255.0
                    if idx == 0:
                        print('------------------------')
                        print('the npz file read are:')
                        print('-------------------------')
                        arr_accumulator = each_npz_arr
                    else:
                        arr_accumulator = np.append (arr_accumulator, each_npz_arr, axis=0)
                    print(each_npz_file)
                    idx = idx + 1
    print()
    print('---------------------------------------------------------------------------')
    print('size of array after reading all the npz files is:', arr_accumulator.shape)
    print('---------------------------------------------------------------------------')
    print()
    return(arr_accumulator)


# Horovod: average metrics from distributed training.
class Metric(object):
# A class for tracking and averaging metrics in distributed training using Horovod.
#
# Attributes:
# name (str): The name of the metric.
# sum (torch.Tensor): The sum of all metric values.
# n (torch.Tensor): The number of metric values added.

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


# save log from rank 0
def save_log_to_h5(args, hvd, loss_acc):
    if hvd.rank() == 0:
        filepath = args.log_file_format
        f = h5py.File(filepath, "w")
        f.create_dataset('train_loss', data=loss_acc['train loss'])
        f.create_dataset('val_loss', data=loss_acc['val loss'])
        f.create_dataset('train_acc', data=loss_acc['train acc'])
        f.create_dataset('val_acc', data=loss_acc['val acc'])

def unfreeze_next_layer(model, layer_names, current_unfreeze_index, hvd):
# Unfreeze the next layer in the model for fine-tuning.
#
# Args:
# model: The model being fine-tuned.
# layer_names (list): List of layer names in the order they should be unfrozen.
# current_unfreeze_index (int): The index of the next layer to unfreeze.
# hvd: Horovod instance.
#
# Returns:
# int: The updated unfreeze index.

    if current_unfreeze_index >= len(layer_names):
        return current_unfreeze_index  # all unfrozen

    layer_to_unfreeze = layer_names[current_unfreeze_index]
    for name, module in model.named_children():
        if name == layer_to_unfreeze:
            for param in module.parameters():
                param.requires_grad = True
            if hvd.rank() == 0: print(f"Unfroze: {layer_to_unfreeze}")
            return current_unfreeze_index + 1  # move to next
    return current_unfreeze_index
