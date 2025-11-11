# Bézier Adversarial Curve Project

---

## 1. Overview
This project provides a **modular and extensible framework** for training, evaluating, and attacking deep neural networks using **Bézier-curve–based adversarial trajectories**.  
It supports flexible combinations of:

- **Datasets:** CIFAR-10 and Mini-ImageNet  
- **Architectures:** ResNet and Vision Transformer (ViT)  
- **Training modes:** pretrained, fine-tune, or from-scratch  

All settings are controlled through a unified `Config` class.  
The framework enables consistent data handling, model management, and adversarial evaluation across diverse experimental setups.

---

## 2. Key Features

### Unified Configuration (`config.py`)
- Central hub controlling dataset, model, training, and attack parameters.  
- Uses enumerations (`DatasetType`, `ModelType`, `TrainingMode`) for safety and extensibility.  
- Dataset-specific mean/std, input size, and transforms managed in one place.  
- Supports automatic resize when ViT is used on CIFAR-10.  
- Centralized PGD/Bézier attack defaults for reproducibility.
#### Example Configuration (CIFAR-10 + ViT):
```bash
DatasetType: CIFAR10
ModelType: ViT
TrainingMode: fine_tune
Image size: 224
Patch size: 16
Embedding dim: 768
Num heads: 12
Num layers: 12
Num labels: 10
PGD steps: 40
Bézier iterations: 30
Learning rate: 0.01
```


---

### Unified Data Pipeline (`data_utils.py`)
- Loads CIFAR-10 and Mini-ImageNet using a common interface.  
- Dynamically composes transforms based on configuration.  
- Automatically resizes images when model requires different size of input (e.g., ViT).  
- Ensures fixed train/test splits for reproducible experiments.

---

### Modular Model Factory (`model_utils.py`)
- Creates ResNet or ViT dynamically according to `Config`.  
- Supports three training modes: pretrained / fine-tune / from-scratch.  
- Automatically replaces classification heads and freezes/unfreezes layers.  
- Unified model saving/loading interface and parameter counting utilities.

---

### Result Management (`results_manager.py`)
- Structured output directory:  
- Automatically saves model checkpoints, logs, and plots.  
- Exports training curves (loss, accuracy, learning rate) and summary statistics.  
- Simplifies reproducibility and paper-ready visualization.

---

### Adversarial Experiments
- PGD attacks (L∞, L2, L1) implemented in `utils.py`.  
- Bézier-curve optimization implemented in `bezier_core.py`.  
- Experiment scripts (`experiment_basic.py`, `experiment_multi_image.py`, `experiment_comprehensive_v2.py`, `experiment_transferability.py`) provide standardized pipelines for evaluating:
- Endpoint attack success,  
- Curve success rate,  
- Rescue and transferability rates.

#### `utils.py`
- Implements standard **Projected Gradient Descent (PGD)** attacks under L∞, L2, and L1 norms.  
- Provides projection and normalization utilities.  
- Evaluates clean/perturbed accuracy for both endpoints and Bézier curves.

#### `bezier_core.py`
- Core implementation of **Bézier-curve adversarial optimization**.  
- Generates smooth perturbation paths connecting multiple adversarial samples.  
- Optimizes curve control points to maximize misclassification along the trajectory.  
- Supports evaluation of success rate, rescue rate, and transferability.

#### `experiment_basic.py`
- Baseline experiment using fixed class settings (A/B/C).  
- Evaluates single-image and two-image Bézier attacks.  
- Logs endpoint and curve-level success rates.

#### `experiment_multi_image.py`
- Multi-image extension of the basic setup.  
- Evaluates performance when using multiple auxiliary images to form Bézier paths.  
- Reports average success rates and variance across images.

#### `experiment_comprehensive_v2.py`
- Comprehensive benchmark covering all A/B/C settings under multiple norms.  
- Supports automatic sample collection, retries, and cross-setting analysis.  
- Used for large-scale evaluation and ablation studies.

#### `experiment_transferability.py`
- Focused on **cross-dataset and cross-model transferability**.  
- Tests whether adversarial endpoints and Bézier paths generated on one model can fool another.  
- Computes metrics such as endpoint transfer success and path rescue ratio.

#### `experiment_basic_v2.py`
- Variant of the basic setup for extended hyperparameter or norm testing.  
- Maintains compatibility with the same configuration interface.

---

## 3. Getting Started

### Install dependencies
```bash
pip install torch torchvision transformers pandas tqdm matplotlib seaborn pillow
```

### CLI — Supported options (concise)

**Parser flags**
- `--dataset` — choose dataset
- `--model`   — choose model architecture
- `--mode`    — choose training mode

### Allowed values

| Flag | Allowed values | Meaning (short) |
|---|---:|---|
| `--dataset` | `cifar10` &#124; `imagenet` | `cifar10` = CIFAR-10; `imagenet` = project shorthand for Mini-ImageNet / ImageNet-style data |
| `--model` | `resnet` &#124; `vit` | `resnet` = ResNet family (default: resnet18) ; `vit` = Vision Transformer (HuggingFace ViT) |
| `--mode` | `pretrained` &#124; `fine_tune` &#124; `from_scratch` | `pretrained` = load weights (no training) ; `fine_tune` = load + fine-tune ; `from_scratch` = random init and train |

### Example: Train a model
```bash
# CIFAR-10 + ResNet (train from scratch)
python train_model.py --dataset cifar10 --model resnet --mode from_scratch

# CIFAR-10 + ViT (fine-tune pretrained model)
python train_model.py --dataset cifar10 --model vit --mode fine_tune

# Mini-ImageNet + ViT (use pretrained weights)
python train_model.py --dataset imagenet --model vit --mode pretrained
```

### Example: Run adversarial experiments
#### Basic Bézier experiment
python experiment_basic.py --dataset cifar10 --model resnet --mode pretrained

#### Comprehensive multi-norm experiment
python experiment_comprehensive_v2.py --dataset cifar10 --model vit --mode fine_tune

#### Multi-image Bézier experiment
python experiment_multi_image.py --dataset imagenet --model vit --mode pretrained

#### Transferability analysis
python experiment_transferability.py --dataset cifar10 --model resnet --mode pretrained


### 4. Project Structure
```bash
├── config.py # Central configuration and enumerations
├── data_utils.py # Unified data loading and transforms
├── model_utils.py # Model creation, loading, and training control
├── train_model.py # General training script
├── results_manager.py # Logging, visualization, and checkpoint handling
├── bezier_core.py # Bézier-curve adversarial optimization
├── utils.py # PGD attacks and evaluation helpers
├── experiment_basic.py # Baseline Bézier experiment
├── experiment_multi_image.py # Multi-image evaluation
├── experiment_comprehensive_v2.py # Comprehensive testing
├── experiment_transferability.py # Transferability analysis
└── results/ # Auto-created experiment outputs
```
