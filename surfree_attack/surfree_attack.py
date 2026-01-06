import torch
from torch.utils.data import TensorDataset, DataLoader
from .surfree import SurFree

def SurFree_AttackWrapper(model, device, dataLoader, config):
    
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