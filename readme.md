# CONCIL: Continual Learning for Multimodal Concept Bottleneck Models

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.x-red.svg)](https://pytorch.org/)

This repository is a **public, reproduction-oriented release** of **CONCIL**.

> **Learning New Concepts, Remembering the Old: Continual Learning for Multimodal Concept Bottleneck Models**

It contains:

- the original project structure and experiment scripts,
- paper figures and visualization assets,
- and a cleaned reproduction entry `reproduce_concil.py` that consolidates local bug fixes found during replication.

If your goal is to **reproduce the project from scratch**, this README is written as a step-by-step guide.

---

## Links

| Paper | Project Page | Repository |
|:---:|:---:|:---:|
| [arXiv PDF](https://arxiv.org/pdf/2411.17471) | [xll0328.github.io/concil](https://xll0328.github.io/concil/) | [GitHub](https://github.com/xll0328/MM25-CONCIL) |

---

## 1. Which code entry should I use?

### Recommended for new users

- `reproduce_concil.py`

Use this first. It is the cleanest public reproduction path in this repo.

### Older/original project scripts

- `src/experiments/CONCIL_1114.py`
- `src/experiments/CONCIL.py`
- `src/experiments/CONCIL_1111.py`
- `command/*.sh`

These are kept for completeness and historical continuity, but for first-time reproduction you should start from `reproduce_concil.py`.

---

## 2. Current paper alignment snapshot

Representative retained local reproduction results are:

| Setting | Metric | Paper | Local reproduction |
|---|---:|---:|---:|
| CUB 2-stage | Avg concept acc | 0.8233 | 0.8237 |
| CUB 2-stage | Avg class acc | 0.6287 | 0.7033 |
| AwA 2-stage | Avg concept acc | 0.9708 | 0.9716 |
| AwA 2-stage | Avg class acc | 0.8739 | 0.8543 |

Interpretation:

- **CUB** is aligned and in the retained run some metrics are higher than the paper table.
- **AwA concept accuracy** is aligned/slightly higher.
- **AwA class accuracy** is slightly below the paper value in the retained run.

So the current public release is best described as:

> a paper-aligned public code release with a cleaned reproduction path.

---

## 3. Environment setup

### 3.1 Create environment

```bash
conda create -n concil python=3.8
conda activate concil
```

### 3.2 Install dependencies

```bash
pip install -r requirements.txt
```

### 3.3 Always run from repository root

```bash
cd MM25-CONCIL
```

---

## 4. Dataset preparation from scratch

This project uses:

- **CUB-200-2011**
- **AwA2 (Animals with Attributes 2)**

A recommended local layout is:

```text
MM25-CONCIL/
├── source_data/
│   ├── CUB_200_2011/
│   └── Animals_with_Attributes2/
└── processed_data/
    ├── cub_processed_data/
    └── awa_processed_data/
```

You do not have to use this exact layout, but it is the easiest one to follow.

### 4.1 Download CUB-200-2011

Place the official dataset under:

```text
source_data/CUB_200_2011
```

The preprocessing script expects the standard CUB structure, including:

- `images/`
- `images.txt`
- `train_test_split.txt`
- `attributes/image_attribute_labels.txt`

### 4.2 Download AwA2

Place the dataset under:

```text
source_data/Animals_with_Attributes2
```

The preprocessing script expects files such as:

- `JPEGImages/`
- `classes.txt`
- `predicate-matrix-binary.txt`

---

## 5. Preprocess the datasets

### 5.1 Preprocess CUB

```bash
python src/processing/cub_data_processing.py \
  -save_dir processed_data/cub_processed_data \
  -data_dir source_data/CUB_200_2011
```

This produces:

- `processed_data/cub_processed_data/train.pkl`
- `processed_data/cub_processed_data/test.pkl`
- `processed_data/cub_processed_data/attribute_map.pkl`

Important note: the CUB preprocessing script filters the original 312 attributes and keeps only attributes with at least 500 positive instances in the training set. That is why this project uses **116 concepts** for CUB.

### 5.2 Preprocess AwA2

```bash
python src/processing/awa_data_processing.py \
  -save_dir processed_data/awa_processed_data \
  -data_dir source_data/Animals_with_Attributes2
```

This produces:

- `processed_data/awa_processed_data/train.pkl`
- `processed_data/awa_processed_data/test.pkl`

Important note: the AwA preprocessing script uses class-level binary predicate vectors and performs a 50/50 random split per class with seed 42.

---

## 6. Configure dataset paths

After preprocessing, create the local path config:

```bash
cp src/utils/data_path.example.yml src/utils/data_path.yml
```

Then edit `src/utils/data_path.yml`.

Example:

```yaml
cub:
  processed_dir: /absolute/path/to/MM25-CONCIL/processed_data/cub_processed_data
  source_dir: /absolute/path/to/MM25-CONCIL/source_data/CUB_200_2011

awa:
  processed_dir: /absolute/path/to/MM25-CONCIL/processed_data/awa_processed_data
  source_dir: /absolute/path/to/MM25-CONCIL/source_data/Animals_with_Attributes2
```

Recommended: use **absolute paths** to avoid path bugs.

---

## 7. Base checkpoint preparation

This is the part most users miss.

### 7.1 Why you need it

`reproduce_concil.py` requires:

- `-base_ckpt /path/to/checkpoint.pth`

CONCIL’s continual stage starts from a pretrained base model, then performs analytic continual updates.

### 7.2 Important fact about this public repository

This public repository **does not currently ship the original pretrained base checkpoints**.

In other words, after cloning this repo you should **not** expect files like these to already exist locally:

- `base_model/CUB/CUB.pth`
- `base_mode_awal/.../AWA.pth`

Those paths existed in the original local development environment, but they are **not included in the public GitHub release**.

### 7.3 What you need to provide

You need a base checkpoint compatible with the chosen dataset:

- CUB run -> CUB base checkpoint
- AwA run -> AwA base checkpoint

### 7.4 If you already have a compatible checkpoint

Use it directly:

```bash
python reproduce_concil.py \
  -dataset cub \
  -base_ckpt /path/to/your/CUB_checkpoint.pth \
  -saved_dir results/concil_repro_cub
```

### 7.5 If you do not have a checkpoint

The repository includes:

- `src/experiments/CONCIL_base_train.py`

But note that this file explicitly says:

> `NOT THE FINAL CODE USED IN THE PROJECT.`

So the most honest and reliable current statement is:

- this repository contains the reference/base-training variant,
- but the cleanest fully supported public path currently assumes that you already have a compatible base checkpoint.

### 7.6 Practical recommendation

If your goal is successful reproduction with the fewest surprises, use this route:

1. preprocess data,
2. prepare or obtain a compatible base checkpoint,
3. run `reproduce_concil.py`,
4. validate the 2-stage result first,
5. then scale to more stages.

### 7.7 What remains to be improved in future cleanup

A future repository cleanup could make reproduction even easier by adding one of these:

- a polished official base-training script,
- released pretrained checkpoints,
- or a separate checkpoint download section.

For the current release, however, the correct expectation is:

> the repo gives you the full continual-learning code path, dataset preprocessing path, and reproduction entry, but **base checkpoints must currently be supplied by the user**.

---

## 8. Run the cleaned reproduction pipeline

### 8.1 Minimal CUB example

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

### 8.2 Minimal AwA example

```bash
python reproduce_concil.py \
  -dataset awa \
  -base_ckpt /path/to/AWA.pth \
  -saved_dir results/concil_repro_awa \
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

## 9. Meaning of the main arguments

- `-dataset`: `cub` or `awa`
- `-base_ckpt`: pretrained base model checkpoint
- `-saved_dir`: output directory
- `-batch_size`: batch size
- `-num_stages`: number of continual phases
- `-class_ratio`: initial visible class ratio in phase 1
- `-concept_ratio`: initial visible concept ratio in phase 1
- `-buffer_size`: expansion buffer size
- `-gg1`: concept-layer regularization
- `-gg2`: class-layer regularization
- `-seed`: random seed
- `-num_workers`: dataloader workers

Paper-style values used in this repo include:

- `class_ratio = 0.5`
- `concept_ratio = 0.5`
- `buffer_size = 25000`
- `gg1 = 500`
- `gg2 = 1`

---

## 10. Output files you should expect

After a successful run, you should see timestamped outputs such as:

- `run.log`
- `overall_summary.csv`
- `stage_1_metrics.csv`
- `stage_2_metrics.csv`
- ...

If these files are produced, your run most likely completed correctly.

---

## 11. How to reproduce multi-stage paper-style experiments

The repo keeps the original shell scripts for multi-stage sweeps.

### CUB sweep

```bash
bash command/CONCIL_cub_exp.sh
```

This loops over:

- `gg1 = 500`
- `gg2 = 1`
- `buffer_size = 25000`
- `num_stages = 2 3 4 5 6 7 8 9 10`

### AwA sweep

```bash
bash command/CONCIL_awa_exp.sh
```

This loops over the same sweep pattern for AwA.

### Hyperparameter sweep

```bash
bash command/CONCIL_tc_11_14.sh
```

This explores multiple values of:

- `gg1`
- `gg2`
- fixed `buffer_size = 25000`
- fixed `num_stages = 3`

---

## 12. Recommended route for first-time users

If you are reproducing this project for the first time, do this exact order:

1. Create environment
2. Install requirements
3. Download raw CUB and/or AwA2
4. Preprocess datasets
5. Copy and edit `src/utils/data_path.yml`
6. Prepare a compatible base checkpoint
7. Run a **2-stage** experiment first
8. Check `overall_summary.csv`
9. Only then try larger stage sweeps

This order avoids most reproduction failures.

---

## 13. Common pitfalls

### Pitfall 1: `src/utils/data_path.yml` does not exist

Fix:

```bash
cp src/utils/data_path.example.yml src/utils/data_path.yml
```

### Pitfall 2: raw dataset folder names do not match expectations

Use:

- `source_data/CUB_200_2011`
- `source_data/Animals_with_Attributes2`

### Pitfall 3: wrong checkpoint path

`reproduce_concil.py` will fail immediately if `-base_ckpt` is wrong.

### Pitfall 4: running outside repository root

Always run commands after:

```bash
cd MM25-CONCIL
```

### Pitfall 5: assuming `CONCIL_base_train.py` is the final polished public base-training script

It is useful as a reference, but not the cleanest supported public entry.

---

## 14. Project structure

```text
.
├── command/                        # Original batch scripts
├── figures/                        # README / paper figures
├── reproduce_concil.py             # Cleaned reproduction entry
├── run_concil_example.sh           # Example run wrapper
├── src/
│   ├── analytic/                   # Recursive analytic modules
│   ├── data/                       # Dataset wrappers and auxiliary CSVs
│   ├── experiments/                # Original project experiment scripts
│   ├── models/
│   ├── processing/                 # Dataset preprocessing scripts
│   └── utils/                      # Config and utility code
├── VISUAL/                         # Visualization assets and notebooks
├── requirements.txt
├── LICENSE
└── readme.md
```

---

## 15. Citation

```bibtex
@inproceedings{lai2025learning,
  title={Learning New Concepts, Remembering the Old: Continual Learning for Multimodal Concept Bottleneck Models},
  author={Lai, Songning and Liao, Mingqian and Hu, Zhangyi and Yang, Jiayu and Chen, Wenshuo and Xiao, Hongru and Tang, Jianheng and Liao, Haicheng and Yue, Yutao},
  booktitle={Proceedings of the ACM International Conference on Multimedia (ACM MM)},
  year={2025}
}
```

---

## 16. License

This project is licensed under the **MIT License**. See `LICENSE` for details.
