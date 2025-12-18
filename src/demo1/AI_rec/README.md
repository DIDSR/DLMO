# AI reconstruction

Demo scripts for AI-based reconstruction methods. A U-Net example is included. The test data and its predictions are large files so does not provided here. To obtain those, please generate them using scripts in synthetic_data_generation folder.

Usage:
```
python synthetic_img_generation.py [-h] [--task TASK] [--test-path TEST_PATH] [--acceleration ACCELERATION]
                                   [--model_name MODEL_NAME] [--num-channels NUM_CHANNELS] [--batch-size BATCH_SIZE]
                                   [--batches-per-allreduce BATCHES_PER_ALLREDUCE] [--fp16-allreduce]
                                   [--pretrained-model-path PRETRAINED_MODEL_PATH]
                                   [--pretrained-model-checkpoint-format PRETRAINED_MODEL_CHECKPOINT_FORMAT]
                                   [--pretrained-model-epoch PRETRAINED_MODEL_EPOCH]

Arguments:
--task: Task type (detection/rayleigh). Default is 'rayleigh'.
--test-path: Path to noisy images for testing.
--acceleration: Acceleration factor (2, 4, 6, or 8).
--model_name: CNN denoiser model (cnn3, redcnn, udncnn, dncnn, unet).
--num-channels: Number of channels (1 for grayscale, 3 for RGB). Default is 1.
--batch-size: Batch size for testing.
--batches-per-allreduce: Number of batches processed locally before allreduce. Default is 1.
--fp16-allreduce: Use fp16 compression during allreduce (flag).
--pretrained-model-path: Path to the directory containing the pre-trained model.
--pretrained-model-checkpoint-format: Format of the checkpoint file. Default is 'checkpoint-{epoch}.pth.tar'.
--pretrained-model-epoch: Epoch number of the pre-trained model to use. Default is 150.
```
