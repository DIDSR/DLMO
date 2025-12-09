
import math
import random

from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
import csv
from guided_diffusion.custom_models import VGG16_custom
import torch
import argparse
import os
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    classifier_and_diffusion_defaults,
    create_classifier_and_diffusion,
)

def create_classifier(num_input_channel,image_size,num_classes,ngf):
    return VGG16_custom( num_input_channel=num_input_channel,image_size=image_size,num_classes=num_classes,ngf=ngf)


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    logger.configure(args.save_dir)
    # logger.configure()

    logger.log("creating model and diffusion...")
    classifier= create_classifier(1,args.image_size,3,32)
    print(classifier)

    classifier.load_state_dict(
        dist_util.load_state_dict(args.classifier_path, map_location="cpu")
    )
    classifier.to(dist_util.dev())
    classifier.eval()

    predictions=np.zeros([args.num_samples,1])
    data_input = np.zeros([args.num_samples,1,args.image_size,args.image_size])
    for i in range(args.num_samples):
        print(i)
        path = args.data_dir+'sample_'+str(i)+'.png'
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
            # if pil_image.size[0] != self.resolution:
            #     pil_image = pil_image.resize((self.resolution,self.resolution))
        arr = np.array(pil_image)
       
        
        
       
        data_input[i,]=arr
    data_input = data_input.astype(np.float32) / 127.5 - 1

    data_input=torch.from_numpy(data_input)

    with torch.no_grad():
        data_input = data_input.to(dist_util.dev())
        logits = classifier(data_input)
        print(logits.shape)
        print(logits)
        _, top_ks = torch.topk(logits, 1, dim=-1)
        print(top_ks)
        # predictions[:,0]=top_ks.cpu().numpy()
    
    labels=np.asarray([0,1,2,1])
    labels=(torch.from_numpy(labels)).to(dist_util.dev())
    print('-------')
    print(logits.shape)
    print(labels.shape)
    a=compute_top_k(logits, labels, k=2, reduction="none")
    print(a)
    print(a.mean().item())
    # np.save(args.save_dir+'/predictions.npy', predictions)

def compute_top_k(logits, labels, k, reduction="mean"):
    _, top_ks = torch.topk(logits, k, dim=-1)
    # print(top_ks)
    if reduction == "mean":
        return (top_ks == labels[:, None]).float().sum(dim=-1).mean().item()
    elif reduction == "none":
        print(top_ks == labels[:, None])
        return (top_ks == labels[:, None]).float().sum(dim=-1)

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=1.0,
        save_dir='./saved_info',
        data_dir=''
    )
    
    defaults.update(classifier_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
