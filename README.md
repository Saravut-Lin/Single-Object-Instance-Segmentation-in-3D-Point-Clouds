<p align="center">
  <h1 align="center">Single-Object Instance Segmentation in 3D Point Clouds</h1>
</p>

<p align="center">
  <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white"></a>
  <a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-1.x%2F2.x-EE4C2C?logo=pytorch&logoColor=white"></a>
  <a href="https://doi.org/10.3390/ai7030096"><img alt="Publication" src="https://img.shields.io/badge/MDPI_AI-Published-green?logo=doi&logoColor=white"></a>
  <a href="https://creativecommons.org/licenses/by/4.0/"><img alt="License" src="https://img.shields.io/badge/License-CC_BY_4.0-lightgrey"></a>
</p>

<p align="center">
  MSc Artificial Intelligence Dissertation — University of Edinburgh
</p>

---

## Overview

This repository consolidates three deep learning architectures evaluated for **single-object instance segmentation in 3D point clouds**, developed as part of a Master of Science dissertation in Artificial Intelligence at the **University of Edinburgh**. Each model is maintained on its own branch with full training, evaluation, and inference code.

The work contributes to the **MiniMarket80** benchmark — a dataset of 1,200 colored point cloud partial views of 80 grocery objects captured with RGB-D cameras — and culminates in the following publication:

> **The MiniMarket80 Dataset for Evaluation of Unique Item Segmentation in Point Clouds**
> Mohamed Sorour, Emma Rattray, Arfa Syahrulfath, Jorge Jaramillo, Saravut Lin, Barbara Webb
> *AI (MDPI)*, Volume 7, Issue 3, Article 96, 2026
> DOI: [10.3390/ai7030096](https://doi.org/10.3390/ai7030096)

---

## Models

### DGCNN — Dynamic Graph CNN

**Branch:** [`dgcnn`](#switching-between-branches)

DGCNN constructs a **dynamic k-nearest-neighbour graph in feature space** at every network layer, applying edge convolutions that capture local geometric structure. Unlike static graph methods, the graph topology is recomputed after each layer, enabling the network to learn evolving point relationships. This implementation adapts DGCNN for binary semantic segmentation (object vs. background) on MiniMarket scenes, employing a dual-loss strategy (weighted cross-entropy + Dice loss) with cosine-annealed learning rate scheduling to handle ~1:9 class imbalance.

### PointWeb

**Branch:** [`pointweb`](#switching-between-branches)

PointWeb (Zhao et al., CVPR 2019) enhances local region representations by constructing a **fully-connected web among points within each neighbourhood**, enabling dense contextual information exchange between all point pairs. This fork adapts PointWeb for binary segmentation on MiniMarket data and additionally includes **model compression pipelines** — structured pruning, post-training quantization, and knowledge distillation — to evaluate deployment efficiency.

### Stratified Transformer

**Branch:** [`stratified-transformer`](#switching-between-branches)

Stratified Transformer applies **transformer-based self-attention** to 3D point clouds using a stratified key-sampling strategy that balances local detail and long-range context. The pipeline includes voxelisation preprocessing and chunk-wise inference with a voting mechanism. This implementation targets binary segmentation on MiniMarket scenes and produces colorised `.ply` output masks.

---

## Model Comparison

| | DGCNN | PointWeb | Stratified Transformer |
|---|:---:|:---:|:---:|
| **Best mIoU** | 0.987 | 0.991 | 0.756 |
| **Object IoU** | 0.974 | 0.975 | 0.601 |
| **Background IoU** | 0.997 | 0.997 | 0.910 |
| **Overall Accuracy** | 99.8% | ≥99.8% | 92.1% |
| **Inference Speed** | — | 1.91 s/scene | 115.17 s/scene |
| **Training Epochs (best)** | 100 (85) | 27 | 70 |
| **Architecture** | Dynamic graph CNN | Point-wise MLP + context web | Transformer + voxelisation |
| **Compression Support** | — | Pruning, Quantization, Distillation | — |

> PointWeb achieves the best balance of accuracy and inference speed. DGCNN closely matches on segmentation quality. Stratified Transformer, while architecturally more expressive, underperforms on this particular binary segmentation task.

---

## Repository Structure

```
main                        ← You are here (README + overview)
├── dgcnn                   ← DGCNN implementation and experiments
├── pointweb                ← PointWeb implementation + compression pipelines
└── stratified-transformer  ← Stratified Transformer implementation
```

Each branch is self-contained with its own code, configs, environment files, and documentation.

---

## Switching Between Branches

```bash
# Clone the repository
git clone https://github.com/Saravut-Lin/Single-Object-Instance-Segmentation-in-3D-Point-Clouds.git
cd Single-Object-Instance-Segmentation-in-3D-Point-Clouds

# Switch to a model branch
git checkout dgcnn
git checkout pointweb
git checkout stratified-transformer
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- CUDA-capable GPU with appropriate NVIDIA drivers
- [Conda](https://docs.conda.io/) (recommended for environment management)

### DGCNN

```bash
git checkout dgcnn
conda env create -f dgcnn_env.yml
conda activate dgcnn
# See branch README for training and evaluation commands
```

### PointWeb

```bash
git checkout pointweb
conda env create -f pointweb_env.yml
conda activate pointweb
# See branch README for training, evaluation, and compression commands
```

### Stratified Transformer

```bash
git checkout stratified-transformer
conda env create -f stratified_env.yml
conda activate stratified
pip install -r requirements.txt
# Note: requires torch-points-kernels (git submodule)
git submodule update --init --recursive
# See branch README for training and evaluation commands
```

---

## Citation

If you use this work, please cite:

```bibtex
@article{sorour2026minimarket80,
  title     = {The MiniMarket80 Dataset for Evaluation of Unique Item Segmentation in Point Clouds},
  author    = {Sorour, Mohamed and Rattray, Emma and Syahrulfath, Arfa and Jaramillo, Jorge and Lin, Saravut and Webb, Barbara},
  journal   = {AI},
  volume    = {7},
  number    = {3},
  pages     = {96},
  year      = {2026},
  publisher = {MDPI},
  doi       = {10.3390/ai7030096}
}
```

---

## Acknowledgements

This work was conducted within the **Insect Robotics Group**, Institute for Perception, Action and Behaviour, School of Informatics, University of Edinburgh. Funded by EPSRC grant EP/V008102/1.

---

<p align="center">
  <sub>University of Edinburgh · School of Informatics · MSc Artificial Intelligence</sub>
</p>
