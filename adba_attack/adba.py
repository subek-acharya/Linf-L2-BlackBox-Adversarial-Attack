import torch
import numpy as np
import copy
import random
import sys
from scipy.stats import norm


# ==================== Helper Classes ====================

class Block:
    """Represent blocks of a picture for hierarchical search."""
    
    def __init__(self, x1, x2):
        self.x1 = x1
        self.x2 = x2
        self.width = x2 - x1 + 1

    def cut_block(self, subnum=2):
        """Cut block into subnum sub-blocks."""
        bs = []
        for i in range(subnum):
            bs.append(copy.deepcopy(self))
 
        for i in range(subnum):
            line1 = self.x1 + (i * (self.x2 - self.x1)) // subnum
            line2 = self.x1 + ((i + 1) * (self.x2 - self.x1)) // subnum
            bs[i].x1 = line1
            if i > 0:
                bs[i].x1 = line1 + 1
            bs[i].x2 = line2
            bs[i].width = bs[i].x2 - bs[i].x1 + 1
        return bs


class PerturbationDirection:
    """Represent a perturbation direction vector."""
    
    def __init__(self, size_channel, size_x, size_y, init_value, device):
        self.size_channel = size_channel
        self.size_x = size_x
        self.size_y = size_y
        self.device = device
        self.pixnum = size_x * size_y * size_channel
        self.adv_v = [init_value for _ in range(self.pixnum)]
        self.Rmax = 1.0
        self.Rmin = 0.0

        # Initialize with random {-1, +1} if init_value is 0
        if init_value == 0:
            for x in range(len(self.adv_v)):
                self.adv_v[x] = random.choice([-1, 1])
    
    def reverse_block(self, block):
        """Reverse signs of perturbation in a block."""
        for x in range(block.x1, block.x2 + 1):
            self.adv_v[x] *= -1

    def to_tensor(self):
        """Convert perturbation direction to tensor."""
        three_d_list = [
            [[self.adv_v[channel * self.size_x * self.size_y + x * self.size_y + y]
              for y in range(self.size_y)]
             for x in range(self.size_x)]
            for channel in range(self.size_channel)]
        
        perturbation = torch.tensor(np.array(three_d_list), dtype=torch.float32)
        return perturbation.to(self.device)


class IterationState:
    """Manage iteration state for ADBA attack."""
    
    def __init__(self, init_direction, offspring_n, device):
        self.offspring_n = offspring_n
        self.device = device
        self.iter_n = 1
        self.offspring_directions = []
        self.chosen_direction = -1
        self.best_direction = copy.deepcopy(init_direction)
        
        for i in range(offspring_n):
            self.offspring_directions.append(copy.deepcopy(init_direction))
            self.offspring_directions[i].Rmax = 1.0
            self.offspring_directions[i].Rmin = 0.0
    
    def mutation(self, model, original_image, label, target_radius, blocks, binary_mode):
        """Create a new generation through mutation."""
        query_count = 0
        
        # Reverse blocks in offspring directions
        for vi in range(self.offspring_n):
            self.offspring_directions[vi].reverse_block(blocks[vi])
            self.offspring_directions[vi].Rmax = self.best_direction.Rmax
            self.offspring_directions[vi].Rmin = 0.0

        # Compare directions using ADB
        query_count += self.compare_directions_using_adb(
            model, original_image, label, target_radius, binary_mode
        )

        # Restore offspring directions
        for vi in range(self.offspring_n):
            self.offspring_directions[vi].reverse_block(blocks[vi])
        
        # Update best direction if a better one was found
        if self.chosen_direction >= 0:
            self.best_direction.reverse_block(blocks[self.chosen_direction])
            self.best_direction.Rmax = self.offspring_directions[self.chosen_direction].Rmax
            self.best_direction.Rmin = self.offspring_directions[self.chosen_direction].Rmin
            
            for vi in range(self.offspring_n):
                self.offspring_directions[vi].reverse_block(blocks[self.chosen_direction])
                self.offspring_directions[vi].Rmax = self.offspring_directions[self.chosen_direction].Rmax
                self.offspring_directions[vi].Rmin = self.offspring_directions[self.chosen_direction].Rmin
        
        self.iter_n += 1
        return query_count
    
    def compare_directions_using_adb(self, model, original_image, label, target_radius, binary_mode):
        """Algorithm 2: Compare Directions Using Approximate Decision Boundary."""
        perturbations = []
        perturbed_images = []
        predictions = []
        query_count = 0
        successful_directions = []
        self.chosen_direction = -1
        
        # Initialize and test all offspring directions at current Rmax
        for i in range(len(self.offspring_directions)):
            perturbations.append(self.offspring_directions[i].to_tensor())
            perturbed_image = torch.clamp(
                original_image + self.best_direction.Rmax * perturbations[i], 
                0.0, 1.0
            )
            perturbed_images.append(perturbed_image)
            
            with torch.no_grad():
                pred = model(perturbed_image).argmax(1)
            predictions.append(pred.cpu())
            query_count += 1
            
            if pred.item() != label.item():
                successful_directions.append(i)
                self.offspring_directions[i].Rmax = self.best_direction.Rmax
                self.chosen_direction = i
            else:
                self.offspring_directions[i].Rmin = self.best_direction.Rmax

        # Return early if no or only one successful direction
        if len(successful_directions) == 0:
            self.chosen_direction = -1
            return query_count
        elif len(successful_directions) == 1:
            self.chosen_direction = successful_directions[0]
            return query_count

        # Binary search to compare multiple successful directions
        low, high = 0, self.best_direction.Rmax
        
        while high - low >= 1e-5:
            # Get next ADB using the chosen method
            adb = get_next_adb(low, high, target_radius, self.best_direction.Rmax, binary_mode)
            
            temp_successful = copy.deepcopy(successful_directions)
            vi = 0
            
            while vi < len(temp_successful):
                perturbed_images[temp_successful[vi]] = torch.clamp(
                    original_image + adb * perturbations[temp_successful[vi]], 
                    0.0, 1.0
                )
                
                with torch.no_grad():
                    pred = model(perturbed_images[temp_successful[vi]]).argmax(1)
                predictions[temp_successful[vi]] = pred.cpu()
                query_count += 1
                
                if pred.item() != label.item():
                    self.offspring_directions[temp_successful[vi]].Rmax = adb
                    self.chosen_direction = temp_successful[vi]
                    
                    if self.offspring_directions[temp_successful[vi]].Rmax <= target_radius:
                        return query_count
                    vi += 1
                else:
                    self.offspring_directions[temp_successful[vi]].Rmin = adb
                    temp_successful.pop(vi)

            if len(temp_successful) == 0:
                low = adb
            elif len(temp_successful) == 1:
                self.chosen_direction = temp_successful[0]
                return query_count
            elif len(temp_successful) >= 2:
                high = adb
                successful_directions = temp_successful
        
        # If directions are close, return the first one
        self.chosen_direction = successful_directions[0]
        return query_count


# ==================== Helper Functions ====================

def get_next_adb(low, high, target_radius, rmax, binary_mode):
    """
    Get next ADB (Approximate Decision Boundary) value.
    
    Args:
        low: Lower bound
        high: Upper bound  
        target_radius: Target epsilon
        rmax: Maximum radius
        binary_mode: 0 for midpoint, 1 for median-based
    """
    if binary_mode == 0:
        # Simple midpoint
        return (low + high) / 2
    else:
        # Median-based (using probability distribution)
        return get_median_adb(low, high, target_radius, rmax)


def get_median_adb(low, high, target_radius, rmax):
    """Calculate median-based ADB using probability distribution."""
    if high <= target_radius:
        return (low + high) / 2
    
    # Use normal distribution approximation
    try:
        # Simplified median calculation
        mid = (low + high) / 2
        
        # Adjust based on target radius
        if mid > target_radius:
            # Bias toward lower values
            weight = min(1.0, target_radius / mid)
            mid = low + weight * (high - low) * 0.5
        
        return mid
    except:
        return (low + high) / 2


def progress_bar(query, iteration, radius):
    """Display progress bar."""
    sys.stdout.write(f'\rQuery: {query:<6} | Iter: {iteration:<4} | R_inf: {radius:.4f}')
    sys.stdout.flush()


# ==================== Main Attack Function ====================

def ADBA_Attack(model, device, original_image, label, epsilon, budget, 
                init_dir=1, offspring_n=2, binary_mode=1):
    """
    ADBA (Approximate Decision Boundary Approach) L-infinity Attack.
    
    Args:
        model: PyTorch model
        device: torch device
        original_image: Original image tensor [1, C, H, W]
        label: True label tensor [1]
        epsilon: Target L-infinity perturbation bound
        budget: Maximum number of queries
        init_dir: Initial direction (1, -1, or 0 for random)
        offspring_n: Number of offspring directions per iteration
        binary_mode: Binary search mode (0: midpoint, 1: median)
    
    Returns:
        adversarial_image: Adversarial image tensor
        success: Boolean indicating if attack succeeded
        query_count: Total number of queries used
        final_radius: Final L-infinity radius achieved
    """
    model.eval()
    
    # Get image dimensions
    _, channels, size_x, size_y = original_image.shape
    pix_num = channels * size_x * size_y
    
    # Initialize perturbation direction
    v0 = PerturbationDirection(channels, size_x, size_y, init_dir, device)
    
    # Initialize blocks for hierarchical search
    b0 = Block(0, pix_num - 1)
    bs1 = b0.cut_block(offspring_n)
    blocks = [bs1]
    
    # Initialize iteration state
    iteration_state = IterationState(v0, offspring_n, device)
    
    # First mutation
    query_count = iteration_state.mutation(
        model, original_image, label, epsilon, blocks[0], binary_mode
    )
    
    iter_num = 1
    block_iter = 0
    
    progress_bar(query_count, iter_num, iteration_state.best_direction.Rmax)
    
    # Main attack loop
    while (query_count < budget) and (iteration_state.best_direction.Rmax > epsilon):
        block_iter += 1
        blocks_i = []
        
        for i, bi in enumerate(blocks[block_iter - 1]):
            blocks_i.extend(bi.cut_block(offspring_n))
            
            query_plus = iteration_state.mutation(
                model, original_image, label, epsilon,
                blocks_i[offspring_n * i : offspring_n * (i + 1)],
                binary_mode
            )
            
            query_count += query_plus
            iter_num += 1
            
            progress_bar(query_count, iter_num, iteration_state.best_direction.Rmax)
            
            if (iteration_state.best_direction.Rmax <= epsilon) or (query_count >= budget):
                break
        
        blocks.append(copy.deepcopy(blocks_i))
    
    print()  # New line after progress bar
    
    # Generate final adversarial image
    final_radius = iteration_state.best_direction.Rmax
    adversarial_v = iteration_state.best_direction.to_tensor()
    adversarial_image = torch.clamp(
        original_image + final_radius * adversarial_v, 
        0.0, 1.0
    )
    
    # Determine success
    success = (final_radius <= epsilon)
    
    return adversarial_image, success, query_count, final_radius