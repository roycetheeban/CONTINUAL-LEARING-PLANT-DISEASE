# Project Setup & Execution Guide

## 1. Environment Setup

### 1.1 Conda Environment Activation
```bash
# Activate the AIOT conda environment (pre-configured locally)
conda activate AIOT
```

### 1.2 Install Required Packages
```bash
# Install dependencies from environment.yml
conda env create -f environment.yml -n AIOT

# Or update existing environment
conda env update -f environment.yml --prune
```

### 1.3 Jupyter Notebook Configuration
If running notebooks in VS Code:
- Press `Ctrl+Shift+P` → "Select Kernel"
- Choose "AIOT" conda environment from the list
- This ensures all dependencies are available in notebook cells

---

## 2. Hardware & GPU Requirements

### 2.1 Check NVIDIA GPU Availability
```bash
# Check NVIDIA GPU status, CUDA version, and memory usage
nvidia-smi

# Check detailed GPU info
nvidia-smi -q
```

### 2.2 Hardware Recommendations
- **GPU**: NVIDIA GPU with CUDA support (8GB+ VRAM recommended for MobileNetV3-Small)
- **RAM**: 16GB+ system RAM
- **Storage**: 100GB+ for datasets and model checkpoints
- **CUDA Toolkit**: Version 11.0+ (check with `nvcc --version`)
- **cuDNN**: 8.0+ (required for deep learning operations)

### 2.3 GPU Activation in Code
Models automatically use GPU if available:
- PyTorch detects CUDA devices automatically
- Training scripts use `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- Monitor GPU usage during training with `nvidia-smi -l 1` (refreshes every 1 second)

---

## 3. Project Structure Overview

```
.
├── data/                          # Organized dataset (already split & segmented)
│   ├── 00_raw_segmented/          # 38 plant disease classes
│   ├── 01_pretrain_26cls/         # Pre-training dataset (26 non-tomato classes)
│   ├── 02_tomato_5cls/            # 5 tomato classes (main study)
│   ├── 03_tomato_new_2cls/        # 2 additional tomato classes (for continual learning)
│   └── 05_metadata/               # Class labels, split info
│
├── experiments/                   # ⭐ MAIN FOLDER - All training code & configurations
│   ├── part1_case1_scratch/       # Case 1: Training from scratch
│   ├── part1_case2_imagenet_finetune/   # Case 2: ImageNet pre-trained transfer
│   ├── part1_case3A_plantvillage_pretrain/  # Case 3: PlantVillage pre-trained
│   ├── part1_case3B_plantvillage_finetune/  # Case 3: Continual learning phase
│   └── Each case contains:
│       ├── code/                  # Training scripts (.py files)
│       ├── configs/               # Configuration files (.yaml)
│       ├── outputs/               # Baseline results
│       └── Continuation/
│           ├── catA_same_classes/ # Continual learning - same classes
│           ├── catB_new_classes/  # Continual learning - new classes
│           └── {method}/outputs/  # EWC, Replay, Isolation, Naive results
│
├── models/                        # ⭐ Pre-trained model weights (copied from experiments)
│   ├── part1_case1_scratch/
│   ├── part1_case2_imagenet/
│   └── part1_case3_plantvillage/
│       └── continuation/cat_{a,b}/{method}/{cycle}/ → model.pth, theta_star.pt
│
├── configs/                       # ⭐ All configuration files (copied from experiments)
│   ├── part1_case1_scratch/
│   ├── part1_case2_imagenet/
│   └── part1_case3_plantvillage/
│       └── continuation/cat_{a,b}/{method}/{cycle}/ → *.yaml
│
├── results/                       # ⭐ All evaluation results & metrics (copied from experiments)
│   ├── part1_baseline/            # Baseline training results
│   ├── part1_case{1,2,3}/         # Continual learning results
│   │   ├── cat_a/                 # Same-class learning results
│   │   └── cat_b/                 # New-class learning results
│   │       └── {method}/{cycle}/
│   │           ├── logs/          # train_log.csv
│   │           ├── metrics/       # metrics.json, fisher_matrix.pt
│   │           ├── figures/       # Training curves, plots
│   │           └── checkpoints/   # Intermediate model checkpoints
│   ├── comparative_analysis/      # Cross-case comparison metrics (optional)
│   └── plots_summary/             # Summary visualizations (optional)
│
├── docs/                          # Documentation & theory
│   ├── flow.md                    # Execution pipeline explanation
│   ├── inout_out flow.md          # Input/output data flow
│   ├── layer_strategy.md          # Layer freezing & adaptation strategy
│   ├── key_knowlege_theory/       # Background theory on continual learning
│   ├── decisions/                 # Design decisions & rationale
│   ├── part_01/                   # Part 1 baseline study details
│   └── study_notes/               # Research notes & analysis
│
├── environment.yml                # Conda environment specification
├── Guide.md                       # This file - setup & execution guide
└── readme.md                      # Project overview & objectives
```

---

## 4. Data Organization

### 4.1 Data Folder Contents
All datasets are **pre-processed, segmented, and split** into train/test/validation:
- **00_raw_segmented/**: Original 38 plant disease classes
- **01_pretrain_26cls/**: Training set for pre-training (non-tomato species)
- **02_tomato_5cls/**: 5 tomato disease classes (main experimental focus)
- **03_tomato_new_2cls/**: Additional tomato classes for continual learning phase
- **05_metadata/**: Class mappings, split information, metadata

**No data preprocessing needed** - ready to use directly in training scripts

---

## 5. Experiments Folder - The Main Hub

### ⭐ **IMPORTANT: experiments/ is the primary folder for all development**

#### 5.1 Understanding the Experiments Structure
```
experiments/
├── part1_case1_scratch/
│   ├── train.py, eval.py          ← Run training here
│   ├── configs/                   ← Modify hyperparameters
│   ├── outputs/                   ← Baseline results generated
│   └── Continuation/
│       ├── catA_same_classes/
│       │   ├── ewc/code/          ← EWC training implementation
│       │   │   └── train_catA_ewc.py
│       │   ├── ewc/configs/       ← EWC hyperparameters
│       │   ├── experience_replay/code/
│       │   └── ...
│       └── catB_new_classes/      ← Similar structure for new classes
```

#### 5.2 How to Modify & Re-run Training

**Step 1: Modify Code or Config**
```bash
# Edit training script (if needed)
vim experiments/part1_case1_scratch/Continuation/catA_same_classes/ewc/code/train_catA_ewc.py

# Edit configuration
vim experiments/part1_case1_scratch/Continuation/catA_same_classes/ewc/configs/catA_ewc_cycle1.yaml
```

**Step 2: Run Training**
```bash
# Navigate to case directory
cd experiments/part1_case1_scratch/Continuation/catA_same_classes/ewc/code/

# Execute training script
python train_catA_ewc.py --config ../configs/catA_ewc_cycle1.yaml
```

**Step 3: Results Auto-Generated**
- New results saved to `outputs/` folder in experiments/
- Metrics, checkpoints, and logs created automatically

**Step 4: (Optional) Copy Updated Results**
```bash
# After training, copy results to root-level results/ folder
Copy-Item "experiments/.../outputs/*" "results/..." -Recurse -Force
```

#### 5.3 All Python Training Files Located in experiments/
- All `.py` scripts for baseline training
- All `.py` scripts for continual learning methods
- Utility functions and helper modules
- **Modification Point**: Make code/config changes here, then re-run to test changes

---

## 6. Models, Configs & Results Structure

### 6.1 Where Everything Comes From
All folders at root level are **COPIES from the experiments folder**:

| Root Folder | Source | Purpose |
|---|---|---|
| `models/` | `experiments/*/outputs/checkpoints/` | Pre-trained model weights (.pth files) |
| `configs/` | `experiments/*/configs/` & `*/code/configs/` | YAML configuration files for reproducibility |
| `results/` | `experiments/*/outputs/` | Training logs, metrics, evaluation results |

### 6.2 Using Pre-trained Models
```python
import torch
from models.part1_case1_scratch.best_model import load_model

# Load Case 1 baseline model
model = torch.load("models/part1_case1_scratch/best_model.pth")

# Load continual learning result (EWC, Cycle 2)
model = torch.load("models/part1_case1_scratch/continuation/cat_a/ewc/cycle2/model.pth")
```

### 6.3 Reproducing Results
```python
import yaml

# Load configuration
with open("configs/part1_case1_scratch/continuation/cat_a/ewc/cycle1/catA_ewc_cycle1.yaml") as f:
    config = yaml.safe_load(f)

# Train with exact same config to reproduce results
# See experiments/ folder for full training script
```

---

## 7. Documentation Reference

### 7.1 Main Documentation Folders (in docs/)
Three key folders for understanding the methodology:

**docs/key_knowlege_theory/** → Background Theory
- Continual learning algorithms (EWC, Replay, Isolation)
- MobileNetV3 architecture explanation
- Fisher Information Matrix concepts

**docs/layer_strategy/** → Technical Approach
- Layer freezing strategy (G1, G2, G3, G4 groups)
- Adaptation approach per continual learning method
- Architecture modification for new classes

**docs/decisions/** → Design Rationale
- Why certain methods were chosen
- Why specific hyperparameters were tuned
- Trade-offs and alternatives considered

**docs/flow.md** → Pipeline Overview
- Step-by-step execution flow
- How data moves through the pipeline
- Checkpoints and intermediate outputs

**docs/study_notes/** → Research Findings
- Performance analysis
- Method comparisons
- Insights and conclusions

### 7.2 Quick Reference
```bash
# Check theory concepts
cat docs/key_knowlege_theory/*.md

# Understand methodology
cat docs/layer_strategy.md

# See execution flow
cat docs/flow.md

# Check design decisions
cat docs/decisions/*.md
```

---

## 8. Quick Start Checklist

- [ ] Activate environment: `conda activate AIOT`
- [ ] Install packages: `conda env update -f environment.yml`
- [ ] Verify GPU: `nvidia-smi`
- [ ] Check data: `ls data/02_tomato_5cls/`
- [ ] Review docs: `cat docs/flow.md`
- [ ] Explore experiments: `ls experiments/part1_case1_scratch/`
- [ ] Load pre-trained model: See section 6.2
- [ ] Run re-training (optional): See section 5.2

---

## 9. Typical Workflow

1. **Review findings**: Check `results/part1_case{1,2,3}/` and documentation
2. **Load pre-trained model**: Use `models/part1_case*/` for inference/fine-tuning
3. **Modify & experiment**: Edit code/configs in `experiments/`
4. **Run training**: Execute scripts in `experiments/*/code/`
5. **Analyze results**: Check `results/` or `experiments/*/outputs/`
6. **Document**: Update `docs/` with findings

---

## 10. Troubleshooting

### GPU Not Detected
```bash
# Verify CUDA/cuDNN
nvidia-smi
nvcc --version
# Update PyTorch if needed
conda install pytorch::pytorch pytorch::pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Missing Dependencies
```bash
# Reinstall from environment.yml
conda env remove -n AIOT
conda env create -f environment.yml -n AIOT
conda activate AIOT
```

### Out of Memory
- Reduce batch size in config files
- Use gradient accumulation
- Check for large batch_size in `.yaml` files in `experiments/*/configs/`

### Cannot Find Data
- Verify data path: `ls data/02_tomato_5cls/train/`
- Check if data splits are present (train/, test/, val/)
- Update data_path in config if needed

---

## Key Takeaways

✅ **experiments/** is the main development hub - all code & configs live here
✅ **data/** is pre-processed and ready to use
✅ **models/**, **configs/**, **results/** are organized copies from experiments/
✅ **docs/** contains essential theory and methodology information
✅ Always work from **experiments/** folder to make changes, then propagate to root folders
✅ GPU/CUDA is auto-detected and used when available
