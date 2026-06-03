# L∞ & L2 Black-Box Adversarial Attacks

A comprehensive PyTorch implementation for evaluating the robustness of deep learning models using L∞-norm and L2-norm constrained black-box adversarial attacks. The framework supports two evaluation modes: **Model Only** (digital clean samples) and **UNet + Model** (scanned bubble samples with UNet denoiser).

## Overview

This project implements four black-box adversarial attacks designed to evaluate model robustness without requiring access to model gradients or internal parameters. The framework supports 9 model architectures and provides a unified interface for running experiments across different attack methods, perturbation budgets, and defense configurations.

**Attack Type:** Black-Box (query-based access only — no gradient information required)

## Evaluation Mode

### Mode 1: Model Only
- Digital Clean Samples → Model → Prediction
- Uses original validation data for attack
- Evaluates model robustness without defense

## Mode 2: UNet + Model
- Scanned Bubble Samples → UNet (Denoiser) → Model → Prediction
- Uses scanned bubble data (post print-scan process)
- Evaluates combined UNet defense + model robustness
- UNet acts as a denoising autoencoder to remove scan artifacts

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

## Epsilon Values Tested

For L∞ attacks (RayS, ADBA, Square):
```text
epsilon = [255/255, 64/255, 32/255, 16/255, 8/255, 4/255]
```

For L2 attack (Surfree):
```text
L2 distance thresholds = [1, 2, 3, 5, 15, 45]
```

# Supported Models

| # | Model | Architecture | Type |
|---|-------|--------------|------|
| 1 | ResNet20-C | Residual Network | CNN |
| 2 | VGG16-C | Deep Sequential CNN | CNN |
| 3 | CaiT-C | Class-Attention in Image Transformers | Transformer |
| 4 | SVM-C | Support Vector Machine | Classical ML |
| 5 | SNN_VGG16-C | Spiking VGG-16 | Spiking Neural Network |
| 6 | SNN_ResNet20-C | Spiking ResNet-20 | Spiking Neural Network |
| 7 | xAI_VGG16-C | Explainable AI (ProtoPNet + VGG-16) | Interpretable |
| 8 | xAI_ResNet20-C | Explainable AI (ProtoPNet + ResNet-20) | Interpretable |
| 9 | MambaVision-L2-C | MambaVision Large-2 | State Space Model |

## Project Structure

```bash
Linf-L2-BlackBoxAttack/
│
├── model_architecture/                # Model implementations
│   ├── ResNet.py                     # ResNet-20 architecture
│   ├── VGG.py                        # VGG-16 architecture
│   ├── cait.py                       # CaiT transformer
│   ├── MultiOutputSVM.py             # SVM wrapper
│   ├── UNet.py                       # UNet denoiser
│   ├── spiking_vgg_voter.py          # Spiking VGG-16
│   └── spiking_resnet_voter.py       # Spiking ResNet-20
│
├── adba_attack/                       # ADBA attack implementation
│   ├── adba.py
│   ├── adba_attack.py
│   └── datatools.py
│
├── rays_attack/                       # RayS attack implementation
│   └── AttackWrappersRayS.py
│
├── square_attack/                     # Square attack implementation
│   ├── square_attack_linf.py
│   ├── square_attack_l2.py
│   └── square_attack_utils.py
│
├── surfree_attack/                    # SurFree attack implementation
│   ├── surfree.py
│   ├── surfree_attack.py
│   └── surfree_utils/
│       ├── attack.py
│       ├── dct.py
│       └── utils.py
│
├── checkpoint/                        # Trained model weights
├── data/                              # Datasets (digital + scanned)
│
├── main.py                            # Main execution script
├── ModelFactory.py                    # Unified model loader
├── constants.py                       # Experiment configurations
├── utils.py                           # Helper functions
│
├── RaysAttackExperiment.py            # RayS - Model Only
├── RaysAttackExperiment_Unet.py       # RayS - UNet + Model
├── ADBAAttackExperiment.py            # ADBA - Model Only
├── ADBAAttackExperiment_Unet.py       # ADBA - UNet + Model
├── SquareAttackLinfExperiment.py      # Square - Model Only
├── SquareAttackLinfExperiment_Unet.py # Square - UNet + Model
├── SurfreeAttackExperiment.py         # SurFree - Model Only
├── SurfreeAttackExperiment_Unet.py    # SurFree - UNet + Model
│
└── README.md
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
SurFree Attack (L2) for all models

### Running Individual Attacks

#### Model Only Mode
```bash
python RaysAttackExperiment.py
python ADBAAttackExperiment.py
python SquareAttackLinfExperiment.py
python SurfreeAttackExperiment.py
```

#### UNet + Model Mode
```bash
python RaysAttackExperiment_Unet.py
python ADBAAttackExperiment_Unet.py
python SquareAttackLinfExperiment_Unet.py
python SurfreeAttackExperiment_Unet.py
```

### Configuration

#### Adding New Models
Use the `ModelFactory` class to load models:
```python
from ModelFactory import ModelFactory

factory = ModelFactory()

# Standard models
model = factory.get_model("resnet20_combined", "checkpoint/model.pth")
model = factory.get_model("mambavision_combined", "checkpoint/mamba_model.pth")

# UNet + Model wrapper
unet_model = factory.get_unet_model_wrapper(
    model_name="resnet20_combined",
    model_checkpoint="checkpoint/model.pth",
    unet_checkpoint="checkpoint/UNet.th",
)
```

#### Configuring Experiments
Edit `constants.py` to define experiment configurations:
```python
# Model Only experiments
EXPERIMENTS_ALL = {
    "resnet20_combined": {
        "ckpt_path": "checkpoint/ModelResNet20.th",
        "dataset_path": "data/kaleel_final_dataset_val_OnlyBubbles_Grayscale.pth",
        "use_unet": False,
    },
}

# UNet + Model experiments
EXPERIMENTS_UNET_ALL = {
    "resnet20_combined": {
        "ckpt_path": "checkpoint/ModelResNet20.th",
        "dataset_path": "data/ResNet-Combined_correct_samples_1000_scanned_bubbles.pt",
        "use_unet": True,
        "unet_ckpt": "checkpoint/UNet.th",
    },
}
```

### Output

```makefile
Results are saved to text files
Adversarial samples are saved to adv_samples/{attack_name}/{model_name}/.
```

### References

## References

- [1] Chen, J., & Gu, Q. (2020). [RayS: A Ray Searching Method for Hard-label Adversarial Attack](https://arxiv.org/abs/2006.12792). *Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, pages 1739–1747.

- [2] Wang, W., Zuo, X., Huang, H., & Chen, G. (2025). [ADBA: Approximation Decision Boundary Approach for Black-Box Adversarial Attacks](https://arxiv.org/abs/2406.04998). *Proceedings of the AAAI Conference on Artificial Intelligence*, pages 7628–7636.

- [3] Andriushchenko, M., Croce, F., Flammarion, N., & Hein, M. (2020). [Square Attack: A Query-Efficient Black-Box Adversarial Attack via Random Search](https://arxiv.org/abs/1912.00049). *European Conference on Computer Vision*, pages 484–501.

- [4] Maho, T., Furon, T., & Le Merrer, E. (2021). [SurFree: A Fast Surrogate-Free Black-Box Attack](https://arxiv.org/abs/2011.12807). *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 10430–10439.

- [5] Ali, H., Patel, M., & Agarwal, N. (2024). [MambaVision: A Hybrid Mamba-Transformer Vision Backbone](https://github.com/NVlabs/MambaVision). *NVIDIA Research*.
