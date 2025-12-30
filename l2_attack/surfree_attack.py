import torch
from torch.utils.data import TensorDataset, DataLoader
from .surfree import SurFree

def SurFree_AttackWrapper(model, device, dataLoader, config=None):
    """
    Alternative wrapper that accepts a configuration dictionary.
    Useful when loading parameters from a JSON config file.
    
    Args:
        model: PyTorch model
        device: torch device
        dataLoader: DataLoader with clean images and labels
        config (dict): Configuration dictionary with 'init' and 'run' keys
    
    Returns:
        advLoader: DataLoader containing adversarial images
    """
    
    # Default configuration
    if config is None:
        config = {
            "init": {
                "steps": 100,
                "max_queries": 5000,
                "BS_gamma": 0.01,
                "BS_max_iteration": 7,
                "theta_max": 30,
                "n_ortho": 100,
                "rho": 0.95,
                "T": 1,
                "with_alpha_line_search": True,
                "with_distance_line_search": False,
                "with_interpolation": False,
                "final_line_search": True,
                "quantification": True,
                "clip": True
            },
            "run": {
                "basis_params": {
                    "basis_type": "dct",
                    "dct_type": "full",
                    "frequence_range": (0, 0.5),
                    "beta": 0.001,
                    "tanh_gamma": 1,
                    "random_noise": "normal"
                }
            }
        }
    
    # Initialize attack with config
    attack = SurFree(**config["init"])
    
    all_adv_images = []
    all_labels = []
    
    total_batches = len(dataLoader)
    
    for batch_idx, (images, labels) in enumerate(dataLoader):
        print(f"\nProcessing batch {batch_idx + 1}/{total_batches}")
        
        images = images.to(device)
        labels = labels.to(device)
        
        # Run attack with run config parameters
        adv_images = attack(model, images, labels, **config["run"])
        
        all_adv_images.append(adv_images.cpu())
        all_labels.append(labels.cpu())
    
    all_adv_images = torch.cat(all_adv_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    adv_dataset = TensorDataset(all_adv_images, all_labels)
    advLoader = DataLoader(adv_dataset, batch_size=dataLoader.batch_size, shuffle=False)
    
    return advLoader