# Evaluating the resolution of AI-based accelerated MR reconstruction using a DLMO
This **DLMO (Deep Learning Model Observer)** implementation evaluates multi-coil sensitivity encoding parallel MR imaging systems at different acceleration factors on the Rayleigh discrimination task as a surrogate measure of resolution, as detailed in [our DLMO paper](https://arxiv.org/abs/2602.22535). You can implement the DLMO approach with other image acquisition settings and reconstruction methods to demonstrate their diagnostic efficacy.

![Evaluation using the DLMO framework](docs/github_fig1.png "Evaluation using the DLMO framework")

## Requirements

Create a new conda environment and install the required packages as follows:
```
git clone https://github.com/DIDSR/DLMO.git
```
The trained models uploaded to this repository total approximately 2.17 GB. In case of bandwidth issues or if you only need to use the code, you can enable GitHub’s smudging option to download the models as pointers instead.
```
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/DIDSR/DLMO.git #to avoid LFS quota issue and download this repo without the trained models
git lfs pull #in case you decide that you will fully download all the models in this repo
```
Create the required environments as appropriate to your need:
```
conda env create --name ddpm --file requirements/ddpm.yml #to generate MR images using trained DDPM. DDPM training implemented using https://github.com/openai/improved-diffusion
conda env create --name dlmo --file requirements/dlmo.yml #to run the dlmo codes
```

## Usage
The example codes below demonstrate how to run the DLMO. Before running the code, update the input paths and adjust the relevant parameters according to your specific application. A demo-by-demo index is also provided in `src/README.md`.

1. [**Object generation using DDPM**](https://github.com/DIDSR/DLMO/tree/main/src/demo1)

    Generate a large batch of MR image samples using a trained DDPM[^refDDPM] model. The generated `.npz` file is the object input for demos 2 and 3.

2. [**Synthetic defect insertion**](https://github.com/DIDSR/DLMO/tree/main/src/demo2)

    Insert singlet and doublet signals into DDPM-generated objects and save the resulting objects in HDF5 format.

3. [**MR acquisition and reconstruction**](https://github.com/DIDSR/DLMO/tree/main/src/demo3)

    Perform forward projection and rSOS reconstruction on DDPM-generated objects to create test datasets in HDF5 format.

4. [**DLMO training**](https://github.com/DIDSR/DLMO/tree/main/src/demo4)

    Train the deep learning-based model observer. This training demo is separated from the small end-to-end example and saves models for later testing.

5. [**A simple example of the DLMO framework**](https://github.com/DIDSR/DLMO/tree/main/src/demo5)

    Run a compact workflow with bundled example for: 1) object generation,  2) image acquisition and reconstruction, 2) AI reconstruction, and 3) DLMO testing.

6. [**Statistical analysis**](https://github.com/DIDSR/DLMO/tree/main/src/demo6)

    Run the MRMC[^refMRMC]-based statistical analysis to ensure DLMO operates at the same level as human experts. This part includes sample size estimation using a pilot study and pivotal-study similarity testing to demonstrate that DLMO and human performance are within a predefined margin of 0.1. It uses source code from the iMRMC package[^refiMRMC]. We have included relevant R files from the package; therefore, pre-installation of the iMRMC library is not recommended. 


## License and Copyright
DLMO is distributed under the MIT license. See [LICENSE](https://github.com/DIDSR/DLMO/tree/main/LICENSE) for more information.

## Citation
If you use the DLMO data or code in your project, please cite its [arXiv paper](https://arxiv.org/abs/2602.22535):
```
@article{yu2026evaluating,
    title={Evaluating the resolution of AI-based accelerated MR reconstruction using a deep learning-based model observer},
    author={Yu, Zitong and Zeng, Rongping and Samuelson, Frank and Kc, Prabhat},
    journal={arXiv preprint arXiv:2602.22535},
    year={2026}
}
```

## Disclaimer

## References
[^refDDPM]: J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” Advances in neural information processing systems, vol. 33, pp. 6840–6851, 2020.
[^refMRMC]: N. A. Obuchowski, “Multireader receiver operating characteristic studies: a comparison of study designs,” Acad. Radiol., vol. 2, no. 8, pp.709–716, 1995.
[^refiMRMC]:FDA/CDRH, “iMRMC: Software for the Statistical Analysis of multi-reader multi-case studies,” RST Catalog, 2022, https://doi.org/10.5281/ZENODO.6628838.
<!--
1. "HCP-Young Adult 2025,” https://www.humanconnectome.org/study/hcp-young-adult/document/hcp-young-adult-2025-release, 2025.
2. K. Li, H. Li, K. J. Myers, and M. A. Anastasio, “Estimating task-based performance bounds for accelerated MRI image reconstruction methods by use of learned-ideal observers,” in Medical Imaging 2025: Image Perception, Observer Performance, and Technology Assessment, vol. 13409. SPIE, 2025, pp. 125–129.
3. 
4. J. I. Tamir, F. Ong, J. Y. Cheng, M. Uecker, and M. Lustig, “Generalized magnetic resonance image reconstruction using the Berkeley advanced reconstruction toolbox,” in ISMRM Workshop on Data Sampling & Image Reconstruction, Sedona, AZ, vol. 7, 2016, p. 8.
5. FDA/CDRH, “iMRMC: Software for the Statistical Analysis of multi-reader multi-case studies,” RST Catalog, 2022, https://doi.org/10.5281/ZENODO.6628838.
-->
