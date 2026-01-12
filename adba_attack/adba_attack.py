import torch
from torch.utils.data import TensorDataset, DataLoader
from .adba import ADBA_Attack

def ADBA_AttackWrapper(model, device, dataLoader, config):
    """
    Wrapper function for ADBA attack to match your existing attack interface.
    
    Args:
        model: PyTorch model
        device: torch device (cuda/cpu)
        dataLoader: DataLoader with (images, labels)
        config: Dictionary with attack configuration
            - epsilon: L-inf perturbation bound (e.g., 0.3)
            - budget: Maximum number of queries per image
            - init_dir: Initial direction (1, -1, or 0 for random)
            - offspring_n: Number of offspring directions (default: 2)
    
    Returns:
        advLoader: DataLoader with adversarial images and labels
    """
    
    # Extract config parameters with defaults
    epsilon = config.get("epsilon", 0.3)
    budget = config.get("budget", 10000)
    init_dir = config.get("init_dir", 1)
    offspring_n = config.get("offspring_n", 2)
    binary_mode = config.get("binary_mode", 1)  # 0: mid, 1: median
    
    all_adv_images = []
    all_labels = []
    all_queries = []
    all_success = []
    
    total_batches = len(dataLoader)
    
    # ADBA processes one image at a time, so we iterate through samples
    sample_idx = 0
    
    for batch_idx, (images, labels) in enumerate(dataLoader):
        print(f"\nProcessing batch {batch_idx + 1}/{total_batches}")
        
        batch_size = images.shape[0]
        
        for i in range(batch_size):
            image = images[i:i+1].to(device)  # Keep batch dimension [1, C, H, W]
            label = labels[i:i+1].to(device)
            
            print(f"  Sample {sample_idx + 1}: ", end="")
            
            # Run ADBA attack on single image
            adv_image, success, queries, final_radius = ADBA_Attack(
                model=model,
                device=device,
                original_image=image,
                label=label,
                epsilon=epsilon,
                budget=budget,
                init_dir=init_dir,
                offspring_n=offspring_n,
                binary_mode=binary_mode
            )
            
            all_adv_images.append(adv_image.cpu())
            all_labels.append(label.cpu())
            all_queries.append(queries)
            all_success.append(success)
            
            status = "SUCCESS" if success else "FAILED"
            print(f"{status} | Queries: {queries} | L-inf: {final_radius:.4f}")
            
            sample_idx += 1
    
    # Combine all adversarial images
    all_adv_images = torch.cat(all_adv_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Print summary
    success_count = sum(all_success)
    total_count = len(all_success)
    avg_queries = sum(all_queries) / len(all_queries) if all_queries else 0
    
    print("\n" + "=" * 60)
    print(f"ADBA Attack Summary:")
    print(f"  - Success Rate: {success_count}/{total_count} ({100*success_count/total_count:.2f}%)")
    print(f"  - Average Queries: {avg_queries:.2f}")
    print("=" * 60)
    
    # Create adversarial DataLoader
    adv_dataset = TensorDataset(all_adv_images, all_labels)
    advLoader = DataLoader(adv_dataset, batch_size=dataLoader.batch_size, shuffle=False)
    
    return advLoader