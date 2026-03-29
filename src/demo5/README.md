# A simple example of the DLMO framework

This demo uploads pretrained weights and DDPM-generated objects(SOMs) uploaded in this repository and walks through previous demos (demo2, demo3) to demostrate the application of DLMO observer model to get the descrimination-based output relative to different MR reconstruction methods (for a use case study of acceleration factor by 4). It contains three parts that should be run in order:

1. [*Synthetic defect insertion*](https://github.com/DIDSR/DLMO/tree/main/src/demo2)

   Insert the signlet and doublet signals corresponding to acceleration factor 4 and store the corresponding MR images into a hdf5 file as:
   ```
   cd src/demo2
   python signal_insertion_test.py 4 0.7 '4,5,6,7,8'
   ```

2. [*MR acquisition and reconstruction*](https://github.com/DIDSR/DLMO/tree/main/src/demo3)

   Read the hdf5 file from step 1 and perform rSOS-based conventional reconstruction at the accelerated rate (that serves as baseline lower bound) and fully sampled rate (which will serve as reference upper bound)
   ```
   cd src/demo3
   python rsos_ddpm_test.py 4
   ```

3. [*AI reconstruction*](https://github.com/DIDSR/DLMO/tree/main/src/demo5/AI_rec)

   Run a AI-based reconstruction, such as the included U-Net pretrained weights, on the accelerated images from step 2.
   ```
   cd src/demo5/AI_rec
   python DL_denoiser_eval.py --task rayleigh --test-path ../../demo3/rsos_rec --acceleration 4 --model_name unet --num-channels 1 --batch-size 10 --pretrained-model-path trained_model
   ```

4. [*DLMO testing*](https://github.com/DIDSR/DLMO/tree/main/src/demo5/DLMO_test)

   Estimate the probability of a doublet signal using the provided trained DLMO checkpoints, either on the rSOS reconstructions from step 1 or on the CNN-denoised outputs from step 2.
   ```
   cd ../DLMO_test
   python dlmo_test_hvd.py --task rayleigh --test-path ../image_acquisition_and_reconstruction/examples/img_w_signal --acceleration 4 --batch-size 6 --pretrained-model-path ./trained_model/mri_cnn_io_acc_4_hvd/hvd_cpts --pretrained-model-epoch 50
   ```

   Demo 4 contains the separate DLMO training workflow if you need to retrain or refine the observer models.