"""
constants.py

Configuration file containing paths and settings for model experiments.

This file defines:
    - CHECKPOINTS: Paths to pre-trained model checkpoint files (.th, .pth)
                   Supported models: ResNet20, VGG16, CaiT, SVM
    
    - VAL_DATASETS: Paths to validation dataset files (.pth)
    
    - TRAIN_DATASETS: Paths to training dataset files (.pth)
    
    - DATASETS: Combined dictionary of all available datasets
    
    - EXPERIMENTS: Model-dataset configurations for running experiments
                   Each entry maps a model name to its checkpoint and dataset paths

Usage:
    from constants import EXPERIMENTS, CHECKPOINTS, DATASETS
    
    # Access experiment config
    config = EXPERIMENTS["resnet20_combined"]
    model_path = config["ckpt_path"]
    data_path = config["dataset_path"]

Note:
    - All paths are relative to the project root directory
    - Grayscale versions of datasets are used by default
    - SVM model has multiple checkpoint files (base + multi-output)
"""

# -------------------------- Checkpoints for all models -----------------------------------------
CHECKPOINTS = {
    "resnet20_combined": "./checkpoint/ModelResNet20-VotingCombined-v2-Grayscale-Run1.th",
    "vgg16_combined": "./checkpoint/ModelVgg16-C2.th",
    "cait_combined": "./checkpoint/ModelCaiT-trCombined-v2-valCombined-v2-Grayscale-Run1.th",
    "svm_combined": [
        "./checkpoint/sklearn_SVM_Combined_v2_Grayscale_Run1/base_pytorch_svm_combined_v2.pth",
        "./checkpoint/sklearn_SVM_Combined_v2_Grayscale_Run1/multi_output_svm_combined_v2.pth",
    ],
}

# ------------------------ Training and Validation datasets ---------------------------------------
VAL_DATASETS = {
    "OnlyBubbles Val": "./data/kaleel_final_dataset_val_OnlyBubbles_Grayscale.pth",
    "Combined Val": "./data/kaleel_final_dataset_val_Combined_Grayscale.pth",
}

TRAIN_DATASETS = {
    "OnlyBubbles Train": "./data/kaleel_final_dataset_train_OnlyBubbles_Grayscale.pth",
    "Combined Train": "./data/kaleel_final_dataset_train_Combined_Grayscale.pth",
}

DATASETS = {**VAL_DATASETS, **TRAIN_DATASETS}

# ------------------------ Model and datasets dictionary -------------------------------------------
EXPERIMENTS = {
    "resnet20_combined": {
        "ckpt_path": CHECKPOINTS["resnet20_combined"],
        "dataset_path": VAL_DATASETS["OnlyBubbles Val"],
    },
    "cait_combined": {
        "ckpt_path": CHECKPOINTS["cait_combined"],
        "dataset_path": VAL_DATASETS["OnlyBubbles Val"],
    },
    "vgg16_combined": {
        "ckpt_path": CHECKPOINTS["vgg16_combined"],
        "dataset_path": VAL_DATASETS["OnlyBubbles Val"],
    },
    "svm_combined": {
        "ckpt_path": CHECKPOINTS["svm_combined"],
        "dataset_path": VAL_DATASETS["OnlyBubbles Val"],
    },
}

