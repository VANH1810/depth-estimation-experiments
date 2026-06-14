<div align="center">

# UniDepth: From Domain-Specific to Universal Depth Estimation

**Group 1: Computer Vision (2526II_INT3412E_1)**  
**University of Engineering and Technology, Vietnam National University (UET-VNU)**

Monocular metric depth estimation on real RGB-D sensor data and synthetic indoor scenes.

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-supported-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">
  <img alt="Hugging Face" src="https://img.shields.io/badge/Hugging%20Face-models-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black">
  <img alt="Computer Vision" src="https://img.shields.io/badge/Computer%20Vision-depth%20estimation-0F766E?style=for-the-badge">
</p>

<p>
  <a href="https://portal.uet.vnu.edu.vn/courses/5787">
    <img alt="Course" src="https://img.shields.io/badge/Course-2526II__INT3412E__1-1D4ED8?style=flat-square">
  </a>
  <a href="https://github.com/lpiccinelli-eth/UniDepth">
    <img alt="UniDepth" src="https://img.shields.io/badge/Model-UniDepth-111827?style=flat-square&logo=github">
  </a>
  <a href="https://github.com/DepthAnything/Depth-Anything-V2">
    <img alt="Depth Anything V2" src="https://img.shields.io/badge/Model-Depth%20Anything%20V2-111827?style=flat-square&logo=github">
  </a>
  <a href="https://github.com/isl-org/ZoeDepth">
    <img alt="ZoeDepth" src="https://img.shields.io/badge/Model-ZoeDepth-111827?style=flat-square&logo=github">
  </a>
  <a href="https://github.com/apple/ml-hypersim">
    <img alt="HyperSim" src="https://img.shields.io/badge/Dataset-Hypersim-7C3AED?style=flat-square&logo=github">
  </a>
</p>

`Zero-Shot Evaluation` · `Metric Depth` · `Resolution Robustness` · `Lighting Robustness`

</div>

---

## 1. Project Team

| Member | Student ID | 
| --- | --- | 
| Tran Quoc Viet Anh | 23021475 | 
| Duong Gia Bao | 23021471 | 
| Vu Nhat Tuong Van | 23021747 | 
| Bui Thu Phuong | 23021667 |

## 2. Overview

This repository studies whether a universal monocular metric depth model can generalize beyond domain-specific benchmarks. The codebase provides a reproducible pipeline for:

- zero-shot metric-depth inference on real and synthetic indoor datasets;
- baseline comparison between UniDepth, ZoeDepth, and Depth Anything V2 Metric Indoor;
- standard depth metrics such as AbsRel, SqRel, RMSE, MAE, and threshold accuracy;
- report figures including metric grids, prediction-vs-ground-truth scatter plots, qualitative samples, and robustness curves;
- robustness analysis under input-resolution changes and photometric brightness perturbations.

The project does not train or fine-tune the evaluated models. It focuses on inference, evaluation, visualization, and report-ready analysis.

## Table of Contents

- [1. Project Team](#1-project-team)
- [2. Overview](#2-overview)
- [3. Datasets](#3-datasets)
  - [3.1. DepthTest](#31-depthtest)
  - [3.2. HyperSim](#32-hypersim)
- [4. Models and Checkpoints](#4-models-and-checkpoints)
- [5. Key Results](#5-key-results)
- [6. Installation](#6-installation)
- [7. Data Preparation](#7-data-preparation)
- [8. Running Experiments](#8-running-experiments)
  - [8.1. Validation and Smoke Run](#81-validation-and-smoke-run)
  - [8.2. Baseline Experiments](#82-baseline-experiments)
  - [8.3. Unified Experiment Runner](#83-unified-experiment-runner)
  - [8.4. Robustness Experiments](#84-robustness-experiments)
  - [8.5. Tables and Figures](#85-tables-and-figures)
- [9. Repository Structure](#9-repository-structure)
- [10. Development Workflow](#10-development-workflow)

## 3. Datasets

### 3.1. DepthTest

Private indoor RGB-D dataset collected by the team using an Intel RealSense D435I camera. It contains office scenes captured at `640 x 480`; depth values are converted to meters during loading. The final evaluation uses four recordings sampled every fifth frame, giving 458 RGB-depth pairs.

### 3.2. HyperSim

Photorealistic synthetic indoor dataset with dense metric depth annotations. This repository uses a local subset of 98 RGB-depth pairs at `1024 x 768`. HyperSim data is not redistributed here; download it from the original authors:

- Project page: [mikeroberts3000.github.io/papers/hypersim](https://mikeroberts3000.github.io/papers/hypersim/)
- Official code/data repository: [apple/ml-hypersim](https://github.com/apple/ml-hypersim)
- Paper: [Hypersim: A Photorealistic Synthetic Dataset for Holistic Indoor Scene Understanding](https://arxiv.org/abs/2011.02523)

Place the downloaded raw scene data under `data/hypersim/raw/`.

## 4. Models and Checkpoints

- UniDepth: [Repo](https://github.com/lpiccinelli-eth/UniDepth), [Paper](https://arxiv.org/abs/2403.18913).
- ZoeDepth: [Repo](https://github.com/isl-org/ZoeDepth), [Paper](https://arxiv.org/abs/2302.12288).
- Depth Anything V2: [Repo](https://github.com/DepthAnything/Depth-Anything-V2), [Paper](https://arxiv.org/abs/2406.09414).

The main report compares raw metric-depth predictions. Relative-depth outputs are only used for supplementary aligned analysis.

## 5. Key Results

The table below summarizes the main raw metric-depth results. No ground-truth scale fitting is applied in the primary comparison.

| Dataset | Model | Protocol | N | AbsRel ↓ | SqRel ↓ | RMSE ↓ | MAE ↓ | δ1 ↑ |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DepthTest | UniDepth | zero-shot | 458 | **0.079** | **0.028** | **0.217** | **0.156** | **0.979** |
| DepthTest | DA-V2 Metric Base | zero-shot | 458 | 0.456 | 0.374 | 0.664 | 0.624 | 0.305 |
| DepthTest | ZoeDepth | zero-shot | 458 | 0.770 | 0.970 | 1.150 | 1.101 | 0.142 |
| HyperSim | UniDepth | zero-shot | 98 | 0.132 | 0.592 | 0.526 | 0.287 | 0.954 |
| HyperSim | ZoeDepth | zero-shot | 98 | 0.245 | 0.181 | 0.536 | 0.483 | 0.627 |
| HyperSim | DA-V2 Metric Base | in-domain reference | 98 | **0.047** | **0.010** | **0.160** | **0.122** | **0.997** |

## 6. Installation

Create and activate a Python environment, then install the official UniDepth package from the bundled `UniDepth/` source directory:

```bash
cd UniDepth
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu118
cd ..
```

Install the repository dependencies:

```bash
pip install -r requirements.txt
```

Core requirements include `torch`, `numpy`, `opencv-python`, `matplotlib`, `h5py`, `pillow`, and `transformers>=4.45.0`.

## 7. Data Preparation

DepthTest should be arranged as:

```text
data/depth_test/
  recording_*/
    color.avi
    camera_parameters.json
    depth_metadata.json
    depth_data/*.npz
```

HyperSim raw scenes should be placed under:

```text
data/hypersim/raw/<scene_id>/
```

Extract a HyperSim scene into the flat format used by the experiment runners:

```bash
python scripts/extract_hypersim.py \
  --scene_root data/hypersim/raw/ai_001_001 \
  --output_root data/hypersim/samples
```

Large local data and generated artifacts are intentionally ignored by Git: raw recordings, `.npy`, `.npz`, model weights, visualizations, and most files under `data/` and `results/`.

## 8. Running Experiments

### 8.1. Validation and Smoke Run

Run compile validation and a small CPU smoke test first:

```bash
python -m compileall src

python -m src.experiments.run_depth_test \
  --model unidepth \
  --data_root data/depth_test \
  --limit 5 \
  --device cpu \
  --overwrite

python -m src.experiments.run_hypersim \
  --model unidepth \
  --data_root data/hypersim/samples \
  --limit 5 \
  --device cpu \
  --overwrite
```

`--limit` is an alias for `--max_samples` in the dataset-specific runners.

### 8.2. Baseline Experiments

DepthTest:

```bash
python -m src.experiments.run_depth_test --model unidepth --device cuda --overwrite
python -m src.experiments.run_depth_test --model zoedepth --device cuda --overwrite
python -m src.experiments.run_depth_test --model depth_anything_v2_metric_indoor_base --device cuda --overwrite
```

HyperSim:

```bash
python -m src.experiments.run_hypersim --model unidepth --device cuda --overwrite
python -m src.experiments.run_hypersim --model zoedepth --device cuda --overwrite
python -m src.experiments.run_hypersim \
  --model depth_anything_v2_metric_indoor_base \
  --device cuda \
  --allow-train-overlap-reference \
  --overwrite
```

`--allow-train-overlap-reference` is required for Depth Anything V2 Metric Indoor on HyperSim because this result is treated as an in-domain/train-overlap reference.

### 8.3. Unified Experiment Runner

Run the primary baseline suite:

```bash
python -m src.experiments.run_all_experiments \
  --datasets depth_test hypersim \
  --models unidepth zoedepth depth_anything_v2_metric_indoor_base \
  --experiments baseline \
  --device cuda \
  --overwrite \
  --allow-train-overlap-reference
```

Run a smaller representative check:

```bash
python -m src.experiments.run_all_experiments \
  --datasets depth_test hypersim \
  --models unidepth zoedepth \
  --experiments baseline \
  --max-samples 5 \
  --device cpu \
  --overwrite
```

In `run_all_experiments`, `--max-samples` is applied per dataset. For example, `--datasets depth_test hypersim --max-samples 5` evaluates up to 5 DepthTest samples and 5 HyperSim samples.

### 8.4. Robustness Experiments

Resolution sensitivity:

```bash
python -m src.experiments.run_all_experiments \
  --datasets depth_test hypersim \
  --models unidepth zoedepth depth_anything_v2_metric_indoor_base \
  --experiments exp2_resolution \
  --resolution-scales 0.5 0.75 1.0 \
  --max-samples 20 \
  --device cuda \
  --overwrite \
  --allow-train-overlap-reference
```

Photometric brightness robustness:

```bash
python -m src.experiments.run_all_experiments \
  --datasets depth_test hypersim \
  --models unidepth zoedepth depth_anything_v2_metric_indoor_base \
  --experiments exp3_lighting \
  --brightness-factors 0.6 0.8 1.0 1.2 1.4 \
  --max-samples 20 \
  --device cuda \
  --overwrite \
  --allow-train-overlap-reference
```

Recommended progression:

1. Smoke run with `--max-samples 5`.
2. Representative robustness run with `--max-samples 20`.
3. Full run by omitting `--max-samples`.

### 8.5. Tables and Figures

Build comparison tables:

```bash
python -m src.experiments.compare_models --overwrite
```

Create report figures:

```bash
python -m src.visualization.make_report_figures \
  --results-root results \
  --datasets depth_test hypersim \
  --output-dir results/figures \
  --figures metric_summary_grid scatter_pred_vs_gt qualitative_samples qualitative_samples_combined \
  --max-points 100000 \
  --num-qualitative-samples 4 \
  --overwrite
```

Create combined robustness plots:

```bash
python -m src.visualization.plot_robustness_combined
```

## 9. Repository Structure

```text
configs/              Dataset-specific inference/evaluation defaults.
data/                 Local datasets and placement guides.
results/              Generated predictions, metrics, comparisons, and figures.
scripts/              Thin command-line utilities for data preparation.
src/                  Main experiment, evaluation, model, and visualization code.
requirements.txt      Python dependencies for the experiment pipeline.
```

### 9.1. Main Source Modules

| Path | Purpose |
| --- | --- |
| `src/common/pipeline.py` | Shared pipeline for sample loading, inference, prediction resizing, metric computation, and output writing. |
| `src/datasets/depth_test.py` | RealSense DepthTest loader. |
| `src/datasets/hypersim.py` | Extracted HyperSim RGB-depth loader. |
| `src/datasets/extract_hypersim.py` | HyperSim HDF5 extraction logic. |
| `src/models/base.py` | Common model-runner interface. |
| `src/models/factory.py` | Model registry used by CLI keys. |
| `src/models/unidepth_runner.py` | UniDepth runner with camera/intrinsics handling. |
| `src/models/zoedepth_runner.py` | ZoeDepth runner through Hugging Face Transformers. |
| `src/models/depth_anything_v2_runner.py` | Depth Anything V2 relative and metric-indoor runners. |
| `src/evaluation/metrics.py` | AbsRel, SqRel, RMSE, MAE, RMSE log, δ1/δ2/δ3. |
| `src/evaluation/alignment.py` | Raw, median-aligned, and scale-shift-aligned evaluation modes. |
| `src/experiments/run_depth_test.py` | DepthTest CLI entry point. |
| `src/experiments/run_hypersim.py` | HyperSim CLI entry point. |
| `src/experiments/run_all_experiments.py` | Unified baseline and robustness runner. |
| `src/experiments/compare_models.py` | Comparison table builder. |
| `src/experiments/exp2_resolution.py` | Resolution sensitivity experiment. |
| `src/experiments/exp3_lighting.py` | Brightness robustness experiment. |
| `src/experiments/robustness_common.py` | Shared robustness evaluation loop. |
| `src/transforms/resolution.py` | RGB resizing and camera-intrinsics scaling. |
| `src/transforms/lighting.py` | Brightness perturbation transform. |
| `src/visualization/make_report_figures.py` | Main report-figure CLI. |
| `src/visualization/plot_robustness_combined.py` | Combined robustness figure generation. |
| `src/utils/` | I/O, path, and visualization helpers. |

## 10. Development Workflow

Before submitting changes:

```bash
python -m compileall src
python -m src.experiments.run_all_experiments --help
python -m src.experiments.compare_models --help
python -m src.visualization.make_report_figures --help
```

Then run at least one CPU smoke experiment for each affected dataset or model. Check generated files under `results/` before committing because large outputs are intentionally ignored or committed separately.
