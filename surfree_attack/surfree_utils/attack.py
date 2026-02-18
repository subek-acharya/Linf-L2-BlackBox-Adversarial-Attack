import torch
import random
import numpy as np
from .utils import atleast_kdim

def get_init_with_noise(model, X, y, max_queries=1000):
    """
    ADBA-style initialization: Find initial adversarial using block mutation.
    """
    device = X.device
    init = X.clone()
    
    with torch.no_grad():
        already_adv = model(X).argmax(1) != y
    
    if already_adv.all():
        print("Already adversarial!")
        return init
    print("Not adversarial in the beginning")
    
    for i in range(len(X)):
        if already_adv[i]:
            continue
        
        x = X[i:i+1]
        label = y[i].item()
        shape = X.shape[1:]
        n_pixels = int(np.prod(shape))
        queries = 0
        
        # Try +1, -1, then block mutation
        for init_val in [1, -1]:
            d = torch.full(shape, init_val, dtype=torch.float32, device=device)
            adv = torch.clamp(x + d.unsqueeze(0), 0, 1)
            queries += 1
            with torch.no_grad():
                if model(adv).argmax(1).item() != label:
                    init[i] = adv.squeeze(0)
                    print(f"  Sample {i}: Found ({queries} queries)")
                    break
        else:
            # Block mutation search
            direction = [1] * n_pixels
            blocks = [(0, n_pixels - 1)]
            found = False
            
            while blocks and queries < max_queries and not found:
                new_blocks = []
                for start, end in blocks:
                    # Flip block
                    test_dir = direction.copy()
                    for j in range(start, end + 1):
                        test_dir[j] *= -1
                    
                    d = torch.tensor(test_dir, dtype=torch.float32, device=device).reshape(shape)
                    adv = torch.clamp(x + d.unsqueeze(0), 0, 1)
                    queries += 1
                    
                    with torch.no_grad():
                        if model(adv).argmax(1).item() != label:
                            init[i] = adv.squeeze(0)
                            print(f"  Sample {i}: Found ({queries} queries)")
                            found = True
                            break
                    
                    # Subdivide
                    if end > start:
                        mid = (start + end) // 2
                        new_blocks.extend([(start, mid), (mid + 1, end)])
                
                blocks = new_blocks
            
            # Random fallback
            while not found and queries < max_queries:
                d = torch.tensor([random.choice([-1, 1]) for _ in range(n_pixels)],
                                dtype=torch.float32, device=device).reshape(shape)
                adv = torch.clamp(x + d.unsqueeze(0), 0, 1)
                queries += 1
                with torch.no_grad():
                    if model(adv).argmax(1).item() != label:
                        init[i] = adv.squeeze(0)
                        print(f"  Sample {i}: Found random ({queries} queries)")
                        found = True
            
            if not found:
                print(f"  Sample {i}: WARNING - Not found ({queries} queries)")

    print("init shape: ", init.shape)
    return init