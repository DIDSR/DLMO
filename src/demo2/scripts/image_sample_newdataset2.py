"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from torchvision import utils
import time

def main():
    args = create_argparser().parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES']='1,8'

    dist_util.setup_dist()
    logger.configure(args.save_dir)
    # logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    print(args.use_ddim)
    all_images = []
    all_labels = []
    previous=0
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        start_time = time.time()
        sample = sample_fn(
            model,
            (args.batch_size, 1, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        print('Time for batch: ',str(time.time() - start_time))
        # utils.save_image(sample,args.save_dir+'sample_'+str(len(all_images))+'.png',normalize=True)
        # sample = ((sample + 1) * 0.21)
        sample_save=((sample + 1) * 0.5)
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        print(dist.get_rank())
        if args.class_cond:
            class_selected=str(classes.cpu().numpy())
            utils.save_image(sample_save,args.save_dir+'sample_'+str(len(all_images))+'_rank_'+str(dist.get_rank())+'_class_label_'+class_selected+'.png',nrow=4)
        else:
            utils.save_image(sample_save,args.save_dir+'sample_'+str(len(all_images))+'_rank_'+str(dist.get_rank())+'.png',nrow=4)
        print(sample.shape)
        print(sample.dtype)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

        if dist.get_rank() == 0 and (len(all_images) * args.batch_size-previous)>40:
            previous = len(all_images) * args.batch_size
            arr = np.concatenate(all_images, axis=0)
            out_path = os.path.join(logger.get_dir(), "current_samples.npz")
            if args.class_cond:
                label_arr = np.concatenate(all_labels, axis=0)
                np.savez(out_path, arr,label_arr)
            else:
                np.savez(out_path, arr)
            

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        save_dir='./saved_info',
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
