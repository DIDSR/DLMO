import math
import random

from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
import csv



def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
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
    all_files = _list_image_files_recursively(data_dir)
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
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
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
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

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def load_data_numpy(
    *, data_dir, batch_size, image_size, class_cond=False, deterministic=False
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
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_numpy_files_recursively(data_dir)
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = NpyDataset(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_numpy_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["npy"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_numpy_files_recursively(full_path))
    return results


class NpyDataset(Dataset):
    def __init__(self, resolution, image_paths, classes=None, shard=0, num_shards=1):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        print(len(self.local_images))

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        np_image = np.load(path)

        # # We are not on a new enough PIL to support the `reducing_gap`
        # # argument, which uses BOX downsampling at powers of two first.
        # # Thus, we do it by hand to improve downsample quality.
        # while min(*pil_image.size) >= 2 * self.resolution:
        #     pil_image = pil_image.resize(
        #         tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        #     )

        # scale = self.resolution / min(*pil_image.size)
        # pil_image = pil_image.resize(
        #     tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        # )

        arr = np.expand_dims(np_image,axis=2)
        # crop_y = (arr.shape[0] - self.resolution) // 2
        # crop_x = (arr.shape[1] - self.resolution) // 2
        # arr = arr[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]
        arr = 2*(arr.astype(np.float32) / 0.42) - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict



def load_data_FDA(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
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
    all_files = _list_image_files_recursively(data_dir)
    print(len(all_files))
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        if '_' in bf.basename(all_files[0]):
            class_names = [bf.basename(path).split("_")[0] for path in all_files]
        elif '-' in bf.basename(all_files[0]):
            class_names = [bf.basename(path).split("-")[0] for path in all_files]
        else:
            raise ValueError("unsupported class seperation")
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        print(sorted_classes)
        classes = [sorted_classes[x] for x in class_names]
    dataset = ImageDataset_FDA(
        image_size,
        all_files,
        classes=classes,
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
    while True:
        yield from loader





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
        print(len(self.local_images))
        # print(self.local_classes.shape)
        print(num_shards)

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

    print(len(label_dict))

    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        for entry2 in sorted(bf.listdir(full_path)):
            full_path2 = bf.join(full_path, entry2)
            ext = entry2.split(".")[-1]
            if "." in entry2 and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
                results.append(full_path2)
                labels.append(label_dict[entry+'/'+entry2])

    print(len(results))
    print(len(labels))
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
    sample_per_class=10000,
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
    print(len(all_files))
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        # class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        print(sorted_classes)
        classes = [sorted_classes[x] for x in class_names]
    #select random 10000 sample from each class
    all_files_new = []
    classes_new = []
    for class_id in range(4):
        sub_class_index=[i for i in range(len(classes)) if classes[i] == class_id]
        print(len(sub_class_index))
        selected = random.sample(range(len(sub_class_index)), sample_per_class)
        all_files_new.extend([all_files[sub_class_index[x]] for x in selected])
        classes_new.extend([classes[sub_class_index[x]] for x in selected])

    print(len(all_files_new))
    sub_class_index=[i for i in range(len(classes_new)) if classes_new[i] == 0]
    print(len(sub_class_index))
    
    dataset = ImageDataset_FDA(
        image_size,
        all_files_new,
        classes=classes_new,
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
    while True:
        yield from loader


def load_data_numpy_single(
    *, data_dir, batch_size, image_size, class_cond=False, deterministic=False
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
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    # all_files = _list_numpy_files_recursively(data_dir)
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = NpyDataset_single(
        image_size,
        data_dir,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader





class NpyDataset_single(Dataset):
    def __init__(self, resolution, data_dir, classes=None, shard=0, num_shards=1):
        super().__init__()
        self.resolution = resolution
        self.all_images = np.load(data_dir)
        self.local_images = self.all_images[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        print(len(self.local_images))

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        np_image = self.local_images[idx,]

        # # We are not on a new enough PIL to support the `reducing_gap`
        # # argument, which uses BOX downsampling at powers of two first.
        # # Thus, we do it by hand to improve downsample quality.
        # while min(*pil_image.size) >= 2 * self.resolution:
        #     pil_image = pil_image.resize(
        #         tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        #     )

        # scale = self.resolution / min(*pil_image.size)
        # pil_image = pil_image.resize(
        #     tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        # )
        if np_image.shape[0]<self.resolution or np_image.shape[1]<self.resolution:
            pad_x1 = (self.resolution-np_image.shape[0])//2
            pad_x2 = self.resolution-np_image.shape[0] - pad_x1
            pad_y1 = (self.resolution-np_image.shape[1])//2
            pad_y2 = self.resolution-np_image.shape[1] - pad_y1
            np_image = np.pad(np_image,((pad_x1,pad_x2),(pad_y1,pad_y2)))

        arr = np.expand_dims(np_image,axis=2)
        # crop_y = (arr.shape[0] - self.resolution) // 2
        # crop_x = (arr.shape[1] - self.resolution) // 2
        # arr = arr[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]
        if arr.max() >1:
            arr = arr.astype(np.float32) / 127.5 - 1
        else:
            arr = arr.astype(np.float32) *2.0 - 1

        # print('-----')
        # print(arr.max())
        # print(arr.min())

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict






def load_data_dat(
    *, data_dir, batch_size, image_size, class_cond=False, deterministic=False
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
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_dat_files_recursively(data_dir)
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = DatDataset(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_dat_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["dat"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_dat_files_recursively(full_path))
    return results


class DatDataset(Dataset):
    def __init__(self, resolution, image_paths, classes=None, shard=0, num_shards=1):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        print(len(self.local_images))

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        np_image = np.fromfile(path,dtype=np.float32,count = -1)
        arr = np.reshape(np_image,(self.resolution,self.resolution,-1))
        # np_image = np.load(path)

        # # We are not on a new enough PIL to support the `reducing_gap`
        # # argument, which uses BOX downsampling at powers of two first.
        # # Thus, we do it by hand to improve downsample quality.
        # while min(*pil_image.size) >= 2 * self.resolution:
        #     pil_image = pil_image.resize(
        #         tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        #     )

        # scale = self.resolution / min(*pil_image.size)
        # pil_image = pil_image.resize(
        #     tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        # )

        # arr = np.expand_dims(np_image,axis=2)
        # crop_y = (arr.shape[0] - self.resolution) // 2
        # crop_x = (arr.shape[1] - self.resolution) // 2
        # arr = arr[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]
        arr = 2*arr - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict

