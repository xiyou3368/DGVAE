Dirichlet Graph Auto-Encoders
============

This is a TensorFlow implementation of the [Dirchlet Graph Variational Auto-Encoder model (DGVAE)](https://arxiv.org/abs/2010.04408), NIPS 2020.

[One PyTorch version is here](https://github.com/DuYooho/DGVAE_pytorch)

DGVAE is an end-to-end trainable neural network model for unsupervised learning, generation and clustering on graphs. This code is more related to graph generation, as described in our paper. 


DGVAE is based on Variational Graph Auto-Encoder (VGAE):

T. N. Kipf, M. Welling, [Variational Graph Auto-Encoders](https://arxiv.org/abs/1611.07308), NIPS Workshop on Bayesian Deep Learning (2016)


## Installation

```bash
python setup.py install
```

## Requirements
* TensorFlow 1.10.0
* python 3.6.4
* networkx
* scikit-learn
* scipy

## Run the demo

```bash
python dgvae/train_generate.py
```

## Model options

--model       default is our_vae(dgvae), others including our_ae(dgae),gcn_vae,gcn_ae,graphite_vae,graphite_ae
--dataset     default is Erdos_Renyi, others including Ego,Regular,Geometric,Power_Law,Barabasi_Albert
