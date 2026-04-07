# L∞ & L2 Black-Box Adversarial Attacks

A comprehensive PyTorch implementation for evaluating the robustness of deep learning models using L∞-norm and L2-norm constrained black-box adversarial attacks.

## Overview

This project implements four black-box adversarial attacks designed to evaluate model robustness without requiring access to model gradients or internal parameters. The framework supports multiple model architectures and provides a unified interface for running experiments across different attack methods and perturbation budgets.

**Attack Type:** Black-Box (query-based access only — no gradient information required)

## What is a Black-Box Attack?

A black-box adversarial attack assumes the attacker has **no knowledge** of the target model's internals:
- No access to model architecture or parameters
- No access to gradients (cannot use backpropagation)
- Only able to query the model and observe outputs (predictions/scores)
- Must rely on query-efficient search strategies

This is a more realistic threat model that reflects real-world attack scenarios where models are deployed as APIs or services.

## Attack Methods

### 1. **RayS Attack (L∞)**
A query-efficient hard-label black-box attack that searches along rays from the original sample. Features:
- Uses binary search along directional rays to find decision boundaries
- Extremely query-efficient for L∞-bounded perturbations
- Hard-label attack (only requires predicted class, not confidence scores)

**Parameters:**
- `epsilon_max (ε)`: Maximum L∞ perturbation magnitude
- `query_limit`: Maximum number of model queries per sample

### 2. **ADBA Attack (L∞)**
Adaptive Direction-based Black-box Attack that iteratively refines perturbation directions. Features:
- Evolutionary strategy with offspring direction sampling
- Binary search for optimal perturbation magnitude
- Adaptive direction refinement based on query feedback

**Parameters:**
- `epsilon (ε)`: Maximum L∞ perturbation magnitude
- `budget`: Maximum query budget per sample
- `init_dir`: Initial direction strategy (0=random, 1=all +1, -1=all -1)
- `offspring_n`: Number of offspring directions per iteration

### 3. **Square Attack (L∞)**
Score-based black-box attack using random square-shaped perturbations. Features:
- Iteratively applies localized square perturbations
- Uses model confidence scores to guide the search
- Highly effective against defended models

**Parameters:**
- `epsilon (ε)`: Maximum L∞ perturbation magnitude
- `n_iters`: Number of attack iterations
- `p_init`: Initial percentage of pixels to perturb
- `loss_type`: Loss function for optimization ("cross_entropy" or "margin")

### 4. **SurFree Attack (L2)**
Surrogate-free decision-based attack optimizing L2 distance. Features:
- Does not require surrogate models
- Uses geometric search in low-dimensional subspaces
- Supports DCT and random basis for perturbation generation
- Tracks samples at multiple L2 distance thresholds

**Parameters:**
- `steps`: Number of optimization steps
- `max_queries`: Maximum queries per image
- `theta_max`: Maximum angle for direction search (degrees)
- `n_ortho`: Number of orthogonal directions to maintain

## Supported Models

| Model | Architecture | Description |
|-------|-------------|-------------|
| **ResNet-20** | Convolutional | Residual network adapted for grayscale images |
| **VGG-16** | Convolutional | Deep CNN with sequential blocks |
| **CaiT** | Transformer | Class-Attention in Image Transformers |
| **MultiOutputSVM** | Classical ML | SVM with PyTorch-compatible interface |

## Project Structure

```bash
Linf-L2-BlackBoxAttack/
│
├── model_architecture/           # Model implementations
│   ├── ResNet.py                # ResNet-20 architecture
│   ├── VGG.py                   # VGG-16 architecture
│   ├── cait.py                  # CaiT transformer
│   └── MultiOutputSVM.py        # SVM wrapper
│
├── adba_attack/                  # ADBA attack implementation
├── rays_attack/                  # RayS attack implementation
├── square_attack/                # Square attack implementation
├── surfree_attack/               # SurFree attack implementation
│
├── checkpoint/                   # Trained model weights
├── data/                         # Dataset storage
│
├── main.py                       # Main execution script
├── ModelFactory.py              # Unified model loader
├── ADBAAttackExperiment.py      # ADBA experiment runner
├── RaysAttackExperiment.py      # RayS experiment runner
├── SquareAttackLinfExperiment.py # Square attack experiment runner
├── SurfreeAttackExperiment.py   # SurFree experiment runner
├── constants.py                  # Experiment configurations
├── utils.py                      # Helper functions
└── README.md                     # This file
```

## Usage

### Running All Attacks
Execute the main script to run all L∞ attacks across multiple epsilon values:
```python
python main.py
```
This will sequentially run:

RayS Attack for all models and epsilon values
ADBA Attack for all models and epsilon values
Square Attack for all models and epsilon values

### Running Individual Attacks

#### RayS Attack
```bash
python RaysAttackExperiment.py
```

#### ADBA Attack
```bash
python ADBAAttackExperiment.py
```

#### Square Attack (L∞)
```bash
python SquareAttackLinfExperiment.py
```

#### SurFree Attack (L2)
```bash
python SurfreeAttackExperiment.py
```

### Configuration

#### Adding New Models
Use the ModelFactory class to load models:
```python
from ModelFactory import ModelFactory
factory = ModelFactory()

# Load ResNet
model = factory.get_model("resnet20", "checkpoint/model.pth")

# Load CaiT
model = factory.get_model("cait", "checkpoint/cait_model.th")

# Load SVM (requires two paths)
model = factory.get_model("svm", ["checkpoint/base.pth", "checkpoint/multi.pth"])
```

#### Configuring Experiments
Edit constants.py to define experiment configurations:
```python
EXPERIMENTS = {
    "resnet20": {
        "ckpt_path": "checkpoint/ModelResNet20.th",
        "dataset_path": "data/dataset.pth"
    },
    "vgg16": {
        "ckpt_path": "checkpoint/ModelVgg16.th",
        "dataset_path": "data/dataset.pth"
    },
    # Add more models...
}
```

### Output
Results are saved to text files with robust accuracy for each model and epsilon value:

```makefile
========== RAYS ATTACK FINAL RESULTS eps=8/255 ==========
resnet20_eps=8/255            : 0.4520
vgg16_eps=8/255               : 0.3890
cait_eps=8/255                : 0.5120
svm_eps=8/255                 : 0.2340
```

```makefile
Adversarial samples are saved to adv_samples/{attack_name}/{model_name}/.
```

### References

- RayS: Chen, J., & Gu, Q. (2020). "RayS: A Ray Searching Method for Hard-label Adversarial Attack"
- ADBA: Based on adaptive direction-based black-box attack methodology
- Square Attack: Andriushchenko, M., et al. (2020). "Square Attack: a query-efficient black-box adversarial attack via random search"
- SurFree: Maho, T., et al. (2021). "SurFree: a fast surrogate-free black-box attack"
