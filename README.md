Contrastive Self-Supervised Learning As Manifold Packing (CLAMP)
---------------------------------------------------------------
This repo contains the implementation for the paper contrastive self-supervised learning as neural manifold packing (NeurIPS2025 maintrack) by [Guanming Zhang](https://scholar.google.com/citations?user=_QRwl9sAAAAJ&hl=en), [David J. Heeger](https://scholar.google.com/citations?user=6ggnUzYAAAAJ&hl=en) and [Stefano Martiniani](https://scholar.google.com/citations?user=pxSj9JkAAAAJ&hl=en).

--------------------
We introduce Contrastive Learning As Manifold Packing (CLAMP), a self-supervised framework that recasts representation learning as a manifold packing problem. CLAMP introduces a loss function inspired by the potential energy of short-range repulsive particle systems, such as those encountered in the physics of simple liquids and jammed packings. In this framework, each class consists of sub-manifolds embedding multiple augmented views of a single image. The sizes and positions of the sub-manifolds are dynamically optimized by following the gradient of a packing loss. This approach yields interpretable dynamics in the embedding space that parallel jamming physics, and introduces geometrically meaningful hyperparameters within the loss function.  

The archetecture is illustrated by the figure below.

<img src="assets/clap_architechture.jpeg" alt="drawing" width="800"/>

The CLAMP framework processes a batch of $b$ input images by applying augmentations to
generate $m$ views for each image. These augmented views are then
encoded and projected into a shared embedding space. Within this
space, the augmented embeddings corresponding to each input form
a distinct sub-manifold, resulting in $b$ such sub-manifolds. Then, a
pairwise packing loss is applied to minimize overlap between these
sub-manifolds. The gradient of the loss is subsequently backpropagated to optimize the model.

A toy example to visualize the training dynamics: we selected 10 images from the
MNIST dataset, one for each digit from 0 to 9, and applied Gaussian noise augmentation. These augmented
images were then encoded into a 3-dimensional embedding space for visualization.
<img src="assets/toy-example.jpg" alt="drawing" width="1000"/>
Following standard linear evaluation protocols, we froze the pretrained ResNet-50 backbone encoder and trained a linear classifier on top of the representation. Training was conducted for 100 epochs on ImageNet-1K and 200 epochs on ImageNet-100. Classification accuracies are reported on the corresponding validation sets
| Method | ImageNet-100 | ImageNet-1K | 1% | 10% |
|:--|:--:|:--:|:--:|:--:|
|     |<div align="center"><b>Linear evaluation</b></div>  | |  **Semi-supervised** | |
| SimCLR  | 79.64 | 66.5 | 42.6 | 61.6 |
| SwAV  | – | **72.1** | **49.8** | **66.9** |
| Barlow Twins  | 80.38* | 68.7 | 45.1 | 61.7 |
| BYOL  | 80.32* | 69.3 | 49.8 | 65.0 |
| VICReg  | 79.4 | 68.7 | 44.75 | 62.16 |
| CorInfoMax  | 80.48 | 69.08 | 44.89 | 64.36 |
| MoCo-V2  | 79.28* | 67.4 | 43.4 | 63.2 |
| SimSiam  | 81.6 | 68.1 | – | – |
| A&U  | 74.6 | 67.69 | – | – |
| MMCR (4 views + ME)  | 82.88 | 71.5 | 49.4 | 66.0 |
| MMCR (8 views) | – | 71.5 | – | – |
| **CLAP (4 views)** | **85.12 ± .05** | 69.50 ± .14 | 47.38 ± .56 | 65.10 ± .30 |
| **CLAP (8 views)** | **85.10 ± .15** | 70.04 ± .16 | 47.87 ± .03 | 65.96 ± .04 |

# Usage
The training and evaluation pipeline consists of three stages:

1 self-supervised pretraining,

2 linear evaluation, and

3 semi-supervised learning.

All training configurations and parameters for these tasks are defined in a single configuration file. Each task section is labeled as [SSL], [LC], or [SemiSL]. Any section that is absent will be skipped during training or evaluation. Note that the [SSL] section is mandatory, as all other tasks require either training or loading the self-supervised model.
## The First step

1 create a folder e.g. mkdir ./test .

2 write the config file(config.ini) inside the folder

3 prepare datasets, for CIFAR10, the data will be automatcally downloaded to ./datasets
  for ImageNet-1K and ImageNet-100, you need to specify the imagenet_train_dir=... and imagenet_val_dir=... in the data session. 
  see  https://github.com/HobbitLong/CMC/blob/master/imagenet100.txt for ImageNet100 labels or use scripts/imagenet100_subset.py for ImageNet100 labels.
## Self-supervised pretraining
Use the following command to run the training:
```
python pretrain.py /path/to/directory /path/to/default_config.ini
```
where we use config files(config_file.ini and default_config.ini) to spcify the training setup and hyperparameters. /path/to/directory must contain the input configureation, config.ini (its name must be config.ini). Missing parameters in your_config.ini will be replaced by its default value specified in default_config.ini. Note that directory to your own imagenet training and validation dataset needs to be specified in either config.ini. Pytorchlightning checkpoints and tensorboard/csv logging file are strored in /path/to/directory .

## Linear evaluation
Once the pretraining is done, use the following command to load the pretrained model and conduct linear evaluation.
```
python linear_probe.py /path/to/pretrained_model /path/to/default_config.ini
```

## Semi-supervised learning
Once the pretraining is done, use the following command to for the semi-supervised learning(only ImageNet-1K is supported).
```
python semi_sl.py /path/to/pretrained_model /path/to/default_config.ini
```
## Examples
See /examples for training setups applied in the paper.
To run the example file for training ImageNet-1K:
```
python pretrain.py /examples/imagenet1k-4views /default_cofigs/default_config_imagenet1k.ini
python linear_probe.py /examples/imagenet1k-4views /default_cofigs/default_config_imagenet1k.ini
python semi_sl.py /examples/imagenet1k-4views /default_cofigs/default_config_imagenet1k.ini
```

## Slurm submission
See greene/submit_batch.sbatch for the slurm submission script.


# Dependencies:
python 3.9 (python 2 is note supported) <br/>
numpy <br/>
scipy <br/>
matplotlib <br/>
lmdb <br/>
pytorch 2.5.1 <br/>
pytrochlightning 2.4.0 <br/>
tensorboard <br/>
albumentation 1.4.24

