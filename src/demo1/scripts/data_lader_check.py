import math
import random

from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
import csv


class ImageDataset_FDA(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip
        # print(len(self.local_images))
        # print(self.local_classes.shape)
        # print(num_shards)

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
            if pil_image.size[0] != self.resolution:
                pil_image = pil_image.resize((self.resolution,self.resolution))

        arr = np.array(pil_image)
        arr = np.expand_dims(arr,axis=-1)
        
        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict





def _list_image_files_FDA(data_dir):
    results = []
    labels = []

    filename='/shared/radon/TOP/mozbey2/challenge_data/breast_types.csv'
    label_dict={}
    with open(filename,'r') as data:
       for ind,line in enumerate(csv.reader(data)):
                #print(line,ind)
                label_dict[line[0]]=line[1]

    # print(len(label_dict))

    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        for entry2 in sorted(bf.listdir(full_path)):
            full_path2 = bf.join(full_path, entry2)
            ext = entry2.split(".")[-1]
            if "." in entry2 and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
                results.append(full_path2)
                labels.append(label_dict[entry+'/'+entry2])

    # print(len(results))
    # print(len(labels))
    return results,labels


def load_data_FDA_class_label(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
    sample_per_class=20000,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files,class_names = _list_image_files_FDA(data_dir)
    # print(len(all_files))
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        # class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        print(sorted_classes)
        classes = [sorted_classes[x] for x in class_names]

        for i in range(len(classes)):
            if classes[i] == 3:
                classes[i]=0
    #select random 10000 sample from each class
    all_files_new_train = []
    classes_new_train = []

    all_files_new_val = []
    classes_new_val = []
    for class_id in range(3):
        sub_class_index=[i for i in range(len(classes)) if classes[i] == class_id]
        # print(len(sub_class_index))
        selected = random.sample(range(len(sub_class_index)), sample_per_class)
        all_files_new_train.extend([all_files[sub_class_index[x]] for x in selected[:int(0.9*sample_per_class)]])
        classes_new_train.extend([classes[sub_class_index[x]] for x in selected[:int(0.9*sample_per_class)]])

        all_files_new_val.extend([all_files[sub_class_index[x]] for x in selected[int(0.9*sample_per_class):]])
        classes_new_val.extend([classes[sub_class_index[x]] for x in selected[int(0.9*sample_per_class):]])

    # print(len(all_files_new_train))
    sub_class_index=[i for i in range(len(classes_new_train)) if classes_new_train[i] == 0]
    # print(len(sub_class_index))

    return all_files_new_train,classes_new_train,all_files_new_val,classes_new_val
    
    # dataset = ImageDataset_FDA(
    #     image_size,
    #     all_files_new,
    #     classes=classes_new,
    #     shard=MPI.COMM_WORLD.Get_rank(),
    #     num_shards=MPI.COMM_WORLD.Get_size(),
    #     random_crop=random_crop,
    #     random_flip=random_flip,
    # )
    # if deterministic:
    #     loader = DataLoader(
    #         dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
    #     )
    # else:
    #     loader = DataLoader(
    #         dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
    #     )
    # while True:
    #     yield from loader


def FDA_data_loader(
    *,
    all_files,
    class_names,
    batch_size,
    image_size,
    random_crop=False,
    random_flip=True,
    deterministic=False,
):
    dataset = ImageDataset_FDA(
        image_size,
        all_files,
        classes=class_names,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    # while True:
    #     yield from loader
    return loader


all_files_new_train,classes_new_train,all_files_new_val,classes_new_val = load_data_FDA_class_label(data_dir='/shared/aristotle/TOP/mozbey2/challenge_data',batch_size=4,image_size=512,class_cond=True)

val_data = FDA_data_loader(all_files=all_files_new_val,class_names=classes_new_val,batch_size=4,image_size=512)

print(len(val_data))
for idx,batch_data in enumerate(val_data):
    if idx%100==0:
        print(idx)
    batch, extra = batch_data[0],batch_data[1] 
    if idx==10:
        print(batch.shape)
        print(extra["y"])
        print(extra["y"].shape)

print(idx)