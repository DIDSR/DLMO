# Conda envirorment creation overview

Denoising Diffusion Probabilistic Models (DDPM)–based object creation and deep learning–based training and inference environments have been separated into `ddpm.yml` and `dlmo.yml`. You may combine them at your own risk.

```
conda env create --name ddpm --file requirements/ddpm.yml #to create MR objects
conda env create --name dlmo --file requirements/dlmo.yml #to run the dlmo testing
```

Our ddpm environment is based on OpenAI’s DDPM code, which can be built separately by following the instructions at:

1. [OpenAI Github](https://github.com/openai/improved-diffusion)

Likewise, our dlmo environment is based on Horovod’s NCCL-based distributed training across multiple GPUs. However, you DO NOT NEED to install Horovod to run the demos provided in this repository. Accordingly, the Horovod package installation line has been commented out in the `dlmo.yml` file.

If you wish to perform multigpu-based training for your AI reconstruction or DLMO trainnig follow the instructions provided in:
    
2. [NCCL build](https://github.com/NVIDIA/nccl)
3. [Horovod installation Guide](https://horovod.readthedocs.io/en/latest/install_include.html)
