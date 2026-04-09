# CONCIL: Continual Learning for Multimodal Concept Bottleneck Models

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.x-red.svg)](https://pytorch.org/)

A cleaned, publication-oriented public release for **CONCIL**, including the original project structure, paper assets, and a **cleaned reproduction script** that consolidates local bug fixes discovered during replication.

> **Learning New Concepts, Remembering the Old: Continual Learning for Multimodal Concept Bottleneck Models**

| [**Paper (PDF)**](https://arxiv.org/pdf/2411.17471) | [**Project Page**](https://xll0328.github.io/concil/) | [**Repository**](https://github.com/xll0328/MM25-CONCIL) |
|:---:|:---:|:---:|
| [arXiv](https://arxiv.org/pdf/2411.17471) | [xll0328.github.io/concil](https://xll0328.github.io/concil/) | [GitHub](https://github.com/xll0328/MM25-CONCIL) |

---

## At a glance

- **Problem:** continual learning for concept bottleneck models when both concepts and classes expand over time
- **Method:** analytic, recursive updates for concept and class layers
- **Code paths:** original experiment scripts + cleaned `reproduce_concil.py`
- **Datasets:** CUB-200-2011 and AwA2
- **Status:** public release is paper-aligned, with representative local reproduction results included in the README discussion

---

## Quick start

### 1. Install dependencies

```bash
conda create -n concil python=3.8
conda activate concil
pip install -r requirements.txt
```

### 2. Configure dataset paths

```bash
cp src/utils/data_path.example.yml src/utils/data_path.yml
```

Then edit `src/utils/data_path.yml` for your local machine.

### 3. Run the cleaned reproduction entry

```bash
python reproduce_concil.py \
  -dataset cub \
  -base_ckpt /path/to/CUB.pth \
  -saved_dir results/concil_repro_cub \
  -batch_size 64 \
  -num_stages 2 \
  -class_ratio 0.5 \
  -concept_ratio 0.5 \
  -buffer_size 25000 \
  -gg1 500 \
  -gg2 1 \
  -seed 42
```

---

## Paper alignment snapshot

| Setting | Metric | Paper | Local reproduction |
|---|---:|---:|---:|
| CUB 2-stage | Avg concept acc | 0.8233 | 0.8237 |
| CUB 2-stage | Avg class acc | 0.6287 | 0.7033 |
| AwA 2-stage | Avg concept acc | 0.9708 | 0.9716 |
| AwA 2-stage | Avg class acc | 0.8739 | 0.8543 |

Interpretation:

- **CUB** is aligned and in the retained run some metrics are higher than the paper table.
- **AwA concept accuracy** is aligned/slightly higher.
- **AwA class accuracy** is slightly below the paper value in the currently retained local run.

---

## Important note on this release

This repository contains:

- the original project structure used for the CONCIL paper codebase,
- the paper figures and experiment scripts,
- and a **cleaned reproduction script** `reproduce_concil.py` that consolidates local fixes made during reproduction.

So this release should be understood as a **cleaned public version of the CONCIL project code and reproduction workflow**, rather than a claim that every file is an untouched historical snapshot.

---

## Table of Contents

- [Overview](#overview)
- [Task: CICIL](#task-cicil)
- [Method: CONCIL](#method-concil)
- [Repository Highlights](#repository-highlights)
- [Paper Alignment and Reproduction Status](#paper-alignment-and-reproduction-status)
- [Environment](#environment)
- [Project Structure](#project-structure)
- [Dataset Preparation](#dataset-preparation)
- [Configuration](#configuration)
- [Training and Evaluation](#training-and-evaluation)
- [Metrics](#metrics)
- [Citation](#citation)
- [License](#license)

---

## Overview

**CONCIL** (**Con**ceptual **C**ontinual **I**ncremental **L**earning) is a continual learning framework for **Concept Bottleneck Models (CBMs)** under a setting where both **classes** and **concepts** expand over time.

It targets **Concept-Incremental and Class-Incremental Continual Learning (CICIL)**:

- new classes arrive over phases,
- new concepts become visible over phases,
- the model must preserve previously learned concept and class knowledge,
- and catastrophic forgetting should be minimized.

Core properties of CONCIL:

- **Gradient-free analytic updates** for concept and decision layers
- **Recursive matrix updates** using only current-phase data and summary statistics
- **No need to retain all historical raw data**
- **Strong stability-plasticity tradeoff** in continual concept/class learning

Datasets used in this project:

- **CUB-200-2011**: 200 classes, 116 concepts used here
- **AwA2**: 50 classes, 85 concepts

---

## Task: CICIL

CONCIL studies **Concept-Incremental and Class-Incremental Continual Learning (CICIL)** for CBMs.

<p align="center">
  <img src="figures/intro-figure.png" width="85%" alt="CICIL task setting" />
</p>
<p align="center"><em>Each phase introduces new classes and expands the accessible concept set.</em></p>

At phase \(t\):

- the model receives current-task data only,
- class space grows,
- concept space grows,
- previous knowledge must be retained.

The CBM includes:

- a concept extractor \(g\),
- and a classifier \(f\),

with both concept dimension and class space expanding over phases.

---

## Method: CONCIL

CONCIL has two major stages:

<p align="center">
  <img src="figures/framework.png" width="95%" alt="CONCIL framework" />
</p>
<p align="center"><em>Base training followed by continual analytic updates.</em></p>

1. **Base training**
   - Jointly train backbone, concept layer, and classifier
   - Freeze the backbone afterward

2. **Continual analytic updates**
   - Update the concept layer via recursive regularized regression
   - Update the classifier via recursive linear regression
   - Expand concept and class dimensions as new tasks arrive

Paper hyperparameters include:

- \(\lambda_1 = 500\)
- \(\lambda_2 = 1\)
- feature expansion dimension \(d_{z^*} = 25000\)
- concept expansion dimension \(d_{\hat{c}^*} = 25000\)

---

## Repository Highlights

This release contains both the original project layout and the cleaned reproduction pathway.

### Main files

- `reproduce_concil.py`: cleaned reproduction script with consolidated fixes
- `src/experiments/CONCIL_1114.py`: original experiment entry used in the project
- `run_concil_example.sh`: minimal example runner
- `command/CONCIL_cub_exp.sh`, `command/CONCIL_awa_exp.sh`: batch scripts

### Visual assets

- `figures/`: README and paper figures
- `VISUAL/`: additional plots and notebooks

### Config

- `src/utils/data_path.example.yml`: example path template
- `src/utils/data_path.yml`: local machine-specific config, kept out of version control

---

## Paper Alignment and Reproduction Status

During local reproduction, a cleaned script `reproduce_concil.py` was created to consolidate bug fixes and path fixes.

The script documents fixes such as:

- local path correction,
- checkpoint path cleanup,
- `concept_ratio` bug fixes,
- metric computation fixes,
- stage-1 forgetting edge-case fix,
- concept prediction evaluation correction.

Representative local reproduction summaries showed the following phase-2 values:

- **CUB 2-stage**: concept accuracy `0.8237`, class accuracy `0.7033`
- **AwA 2-stage**: concept accuracy `0.9716`, class accuracy `0.8543`

Compared with the paper table:

- **CUB reproduction is aligned and in some metrics higher**
- **AwA concept accuracy is aligned/slightly higher**
- **AwA class accuracy is slightly below the paper number in the currently retained local run**

Therefore, the current public release should be described as:

> a paper-aligned CONCIL code release with a cleaned reproduction path and representative local replication results.

---

## Environment

Recommended environment:

- Python 3.8+
- PyTorch
- torchvision
- transformers
- PyYAML
- tqdm
- matplotlib
- Pillow

Install:

```bash
conda create -n concil python=3.8
conda activate concil
pip install -r requirements.txt
```

Run commands from the repository root.

---

## Project Structure

```text
.
├── command/                        # Experiment shell scripts
├── figures/                        # README/paper figures
├── reproduce_concil.py             # Cleaned reproduction script
├── run_concil_example.sh           # Example run entry
├── src/
│   ├── analytic/                   # Recursive analytic modules
│   ├── data/                       # Dataset wrappers and auxiliary CSVs
│   ├── experiments/                # Original project experiment scripts
│   ├── models/
│   ├── processing/                 # Dataset preprocessing
│   └── utils/                      # Config and utilities
├── VISUAL/                         # Additional plots and notebooks
├── requirements.txt
├── LICENSE
└── readme.md
```

---

## Dataset Preparation

1. Download datasets:
   - **CUB-200-2011**
   - **AwA2**

2. Preprocess from repository root:

```bash
python src/processing/cub_data_processing.py \
  -save_dir processed_data/cub_processed_data \
  -data_dir source_data/CUB_200_2011

python src/processing/awa_data_processing.py \
  -save_dir processed_data/awa_processed_data \
  -data_dir source_data/Animals_with_Attributes2
```

This produces processed dataset files used by the experiment scripts.

---

## Configuration

Copy the example config first:

```bash
cp src/utils/data_path.example.yml src/utils/data_path.yml
```

Then edit dataset paths for your own machine.

Example entries are provided for:

- `cub`
- `awa`
- `cebab`
- `imdb`

---

## Training and Evaluation

> **Recommended entry for new users:** start from `reproduce_concil.py`.  
> The older scripts under `src/experiments/` are preserved for completeness and historical continuity.

### Recommended cleaned reproduction entry

```bash
python reproduce_concil.py \
  -dataset cub \
  -base_ckpt /path/to/CUB.pth \
  -saved_dir results/concil_repro_cub \
  -batch_size 64 \
  -num_stages 2 \
  -class_ratio 0.5 \
  -concept_ratio 0.5 \
  -buffer_size 25000 \
  -gg1 500 \
  -gg2 1 \
  -seed 42
```

### Example runner

```bash
bash run_concil_example.sh
```

### Original experiment script path

```bash
python src/experiments/CONCIL_1114.py \
  -dataset cub \
  -num_stages 8 \
  -buffer_size 25000 \
  -saved_dir results/concil_cub
```

### Batch scripts

```bash
bash command/CONCIL_cub_exp.sh
bash command/CONCIL_awa_exp.sh
bash command/CONCIL_tc_11_14.sh
```

---

## Metrics

The project reports four main continual learning metrics:

- **Average concept accuracy**
- **Average class accuracy**
- **Average concept forgetting rate**
- **Average class forgetting rate**

Higher accuracies and lower forgetting rates indicate better continual performance.

---

## Citation

```bibtex
@inproceedings{lai2025learning,
  title={Learning New Concepts, Remembering the Old: Continual Learning for Multimodal Concept Bottleneck Models},
  author={Lai, Songning and Liao, Mingqian and Hu, Zhangyi and Yang, Jiayu and Chen, Wenshuo and Xiao, Hongru and Tang, Jianheng and Liao, Haicheng and Yue, Yutao},
  booktitle={Proceedings of the ACM International Conference on Multimedia (ACM MM)},
  year={2025}
}
```

---

## License

This project is licensed under the **MIT License**. See `LICENSE` for details.
