# Demo overview

This folder contains the demo workflow described in our paper, [Evaluating the resolution of AI-based accelerated MR reconstruction using a deep learning-based model observer](https://arxiv.org/abs/2602.22535).

The demos are organized so that each stage has a clear role in the workflow:

1. [Demo 1: Object generation using DDPM](demo1/README.md)

   Generate DDPM-based MR object samples and save them as `.npz` files.

2. [Demo 2: Synthetic defect insertion](demo2/README.md)

   Insert singlet and doublet defects into generated objects.

3. [Demo 3: MR acquisition and reconstruction](demo3/README.md)

   Simulate accelerated acquisition and produce rSOS reconstructions.

4. [Demo 4: DLMO training](demo4/README.md)

   Train the model observer with prepared training and validation datasets.

5. [Demo 5: A simple example of the DLMO framework](demo5/README.md)

   Run a compact end-to-end example using bundled sample objects and provided checkpoints.

6. [Demo 6: Statistical analysis](demo6/README.md)

   Run the power-analysis and pivotal-study scripts used in the paper.

Please refer to each demo README for prerequisites, inputs, outputs, and example commands.