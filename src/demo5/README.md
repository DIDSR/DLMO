# A simple example of the DLMO framework

This demo is a compact workflow that can generate example results locally without requiring the full training pipeline. It contains three parts that should be run in order:

1. [*Image acquisition and reconstruction*](https://github.com/DIDSR/DLMO/tree/main/src/demo5/image_acquisition_and_reconstruction)

   Create a small set of accelerated MR examples from bundled DDPM-generated objects. This step writes HDF5 files and optional PNG previews to `image_acquisition_and_reconstruction/examples/img_w_signal/`.

2. [*AI reconstruction*](https://github.com/DIDSR/DLMO/tree/main/src/demo5/AI_rec)

   Run a CNN denoiser, such as the included U-Net example, on the accelerated images from step 1.

3. [*DLMO testing*](https://github.com/DIDSR/DLMO/tree/main/src/demo5/DLMO_test)

   Estimate the probability of a doublet signal using the provided trained DLMO checkpoints, either on the rSOS reconstructions from step 1 or on the CNN-denoised outputs from step 2.

Recommended order:

```
cd src/demo5/image_acquisition_and_reconstruction
python synthetic_img_acquisition.py 4 10 1

cd ../AI_rec
python DL_denoiser_pred.py --task rayleigh --test-path ../image_acquisition_and_reconstruction/examples/img_w_signal --acceleration 4 --model_name unet --num-channels 1 --batch-size 2 --pretrained-model-path trained_model

cd ../DLMO_test
python dlmo_test_hvd.py --task rayleigh --test-path ../image_acquisition_and_reconstruction/examples/img_w_signal --acceleration 4 --batch-size 6 --pretrained-model-path ./trained_model/mri_cnn_io_acc_4_hvd/hvd_cpts --pretrained-model-epoch 50
```

Demo 4 contains the separate DLMO training workflow if you need to retrain or refine the observer models.