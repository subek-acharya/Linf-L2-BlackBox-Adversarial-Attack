import torch
from torch.utils.data import TensorDataset, DataLoader
from .surfree import SurFree, distance  # Import distance from surfree.py


def SurFree_AttackWrapper(model, device, dataLoader, config):
    
    # Initialize attack with config
    attack = SurFree(**config["init"])
    
    all_clean_images = []
    all_adv_images = []
    all_labels = []
    
    total_batches = len(dataLoader)
    
    for batch_idx, (images, labels) in enumerate(dataLoader):
        print(f"\nProcessing batch {batch_idx + 1}/{total_batches}")
        
        images = images.to(device)
        labels = labels.to(device)
        
        # Run attack with run config parameters
        adv_images = attack(model, images, labels, **config["run"])
        
        all_clean_images.append(images.cpu())
        all_adv_images.append(adv_images.cpu())
        all_labels.append(labels.cpu())
    
    # Concatenate all batches
    x_clean = torch.cat(all_clean_images, dim=0)
    x_adv = torch.cat(all_adv_images, dim=0)
    y_labels = torch.cat(all_labels, dim=0)
    
    # Calculate L2 distance for each sample
    l2_distances = distance(x_clean, x_adv)

    # Check if any L2 distance exceeds 45
    max_distance = l2_distances.max().item()
    if max_distance > 45:
        raise ValueError(f"L2 distance exceeds maximum threshold! Max distance: {max_distance:.4f} > 45")
    
    # Define L2 distance thresholds
    thresholds = [1, 2, 3, 5, 15, 45]
    
    # Create 6 dictionaries based on thresholds
    adv_blobs = {}
    
    for threshold in thresholds:
        # Create mask for samples with L2 distance <= threshold
        mask = l2_distances <= threshold
        
        # Filter samples based on mask
        filtered_x_clean = x_clean[mask]
        filtered_x_adv = x_adv[mask]
        filtered_labels = y_labels[mask]
        filtered_l2_distances = l2_distances[mask]
        
        # Create dictionary for this threshold
        adv_blobs[threshold] = {
            "x_clean": filtered_x_clean.float(),
            "x_adv": filtered_x_adv.float(),
            "labels": filtered_labels.long(),
            "l2_distances": filtered_l2_distances.float(),
            "threshold": threshold
        }

        # Print statistics for this threshold
        print(f"\n--- Threshold L2 <= {threshold} ---")
        print(f"  Number of samples: {filtered_x_clean.shape[0]}")
        if filtered_l2_distances.numel() > 0:
            print(f"  L2 distance - Mean: {filtered_l2_distances.mean():.4f}, Min: {filtered_l2_distances.min():.4f}, Max: {filtered_l2_distances.max():.4f}")
        else:
            print(f"  No samples found for this threshold!")
        
    # Create DataLoader for backward compatibility (using all samples)
    adv_dataset = TensorDataset(x_adv, y_labels)
    advLoader = DataLoader(adv_dataset, batch_size=dataLoader.batch_size, shuffle=False)
    
    return advLoader, adv_blobs
