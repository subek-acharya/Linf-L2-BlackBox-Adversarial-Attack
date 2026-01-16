# coding:utf-8
"""
ADBA (Approximation Decision Boundary Approach) Attack Implementation
Adapted from: https://github.com/BUPTAIOC/ADBA

Paper: ADBA: Approximation Decision Boundary Approach for Black-Box Adversarial Attacks
       (AAAI 2025)
"""

import torch
import numpy as np
import copy
import random
import sys

from . import datatools


# ==================== Classes ====================

class Block:
    """Represent blocks of a picture."""
    
    def __init__(self, x1, x2):
        self.x1 = x1
        self.x2 = x2
        self.width = x2 - x1 + 1

    def cut_block(self, subnum=2):
        bs = []
        for i in range(subnum):
            bs.append(copy.deepcopy(self))
 
        for i in range(subnum):
            line1 = self.x1 + (i * (self.x2 - self.x1)) // subnum  # d1
            line2 = self.x1 + ((i + 1) * (self.x2 - self.x1)) // subnum  # d2
            bs[i].x1 = line1
            if i > 0:
                bs[i].x1 = line1 + 1
            bs[i].x2 = line2
            bs[i].width = bs[i].x2 - bs[i].x1 + 1
        return bs


class V:
    """Represent a perturbation direction."""
    
    def __init__(self, size_channel, size_x, size_y, v, device):
        self.size_channel = size_channel
        self.size_x = size_x
        self.size_y = size_y
        self.device = device
        self.pixnum = size_x * size_y * size_channel
        self.adv_v = [v for _ in range(self.pixnum)]
        self.score = 1.0
        self.Rmax = 1.0
        self.Rmin = 0.0

        list_temp = [-1, 1]
        if v == 0:
            for x in range(len(self.adv_v)):
                self.adv_v[x] = random.choice(list_temp)
    
    def reverse_v(self, block):
        """Reverse blocks of a perturbation direction to generate new directions."""
        for x in range(block.x1, block.x2 + 1):
            self.adv_v[x] *= -1

    def advv_to_tensor(self):
        """Convert perturbation direction to tensor."""
        three_d_list = [
            [[self.adv_v[channel * self.size_x * self.size_y + x * self.size_y + y]
              for y in range(self.size_y)]
             for x in range(self.size_x)]
            for channel in range(self.size_channel)]
        aim_np = np.array(three_d_list)
        perturbation = torch.tensor(aim_np, dtype=torch.float32)
        return perturbation.to(self.device)


class Iter:
    """Represent Iterations."""
    
    def __init__(self, init_vbest, offspringN, device, iter_n=1):
        self.offspringN = offspringN
        self.device = device
        self.iter_n = iter_n
        self.offspringVs = []  # d1 d2
        self.chosen_v = -1  # chose which d
        self.old_vbest = copy.deepcopy(init_vbest)  # dbest
        for i in range(offspringN):
            self.offspringVs.append(copy.deepcopy(init_vbest))
            self.offspringVs[i].Rmax, self.offspringVs[i].Rmin = 1.0, 0.0
    
    def mutation(self, model, original_image, label, aim_r, tolerance_binary_iters, blocks, binaryM):
        """Create a new generation."""
        query = 0
        for vi in range(self.offspringN):
            self.offspringVs[vi].reverse_v(blocks[vi])
            self.offspringVs[vi].Rmax, self.offspringVs[vi].Rmin = self.old_vbest.Rmax, 0.0

        query = query + self.compare_directions_usingADB(
            model, original_image, label, aim_r, tolerance_binary_iters, binaryM)

        for vi in range(self.offspringN):  # initialize directions
            self.offspringVs[vi].reverse_v(blocks[vi])
        if self.chosen_v >= 0:
            self.old_vbest.reverse_v(blocks[self.chosen_v])
            self.old_vbest.Rmax, self.old_vbest.Rmin = (
                self.offspringVs[self.chosen_v].Rmax, self.offspringVs[self.chosen_v].Rmin)
            for vi in range(self.offspringN):
                self.offspringVs[vi].reverse_v(blocks[self.chosen_v])
                self.offspringVs[vi].Rmax, self.offspringVs[vi].Rmin = (
                    self.offspringVs[self.chosen_v].Rmax, self.offspringVs[self.chosen_v].Rmin)
        self.iter_n = self.iter_n + 1
        return query
    
    def compare_directions_usingADB(self, model, original_image, label, aim_r, maxIters, binaryM):
        """Algorithm 2: Compare Directions Using ADB."""
        perturbations = []  # d1 d2
        perturbed_images = []  # = x+ADB*d
        predicted = []  # = F(x+ADB*d)
        query = 0
        succV = []
        self.chosen_v = -1
        
        # Initialize directions
        for i in range(len(self.offspringVs)):
            perturbations.append(self.offspringVs[i].advv_to_tensor())
            perturbed_images.append(torch.clamp(
                original_image + self.old_vbest.Rmax * perturbations[i], 0.0, 1.0))
            
            with torch.no_grad():
                pred = model(perturbed_images[i]).argmax(1)
            predicted.append(pred.cpu())
            query = query + 1
            
            if predicted[i].item() != label.item():
                succV.append(i)
                self.offspringVs[i].Rmax = self.old_vbest.Rmax
                self.chosen_v = i
            else:
                self.offspringVs[i].Rmin = self.old_vbest.Rmax

        if len(succV) == 0:
            self.chosen_v = -1
            return query
        elif len(succV) == 1:
            self.chosen_v = succV[0]
            return query

        low, high = 0, self.old_vbest.Rmax
        while high - low >= 1e-5:  # comparison loop
            ADB = datatools.next_ADB(low, high, aim_r, self.old_vbest.Rmax, binaryM)  # guess next ADB using rho(r)
            succVtemp = copy.deepcopy(succV)
            vi = 0
            while vi < len(succVtemp):
                perturbed_images[succVtemp[vi]] = torch.clamp(
                    original_image + ADB * perturbations[succVtemp[vi]], 0.0, 1.0)
                
                with torch.no_grad():
                    pred = model(perturbed_images[succVtemp[vi]]).argmax(1)
                predicted[succVtemp[vi]] = pred.cpu()
                query = query + 1
                
                if predicted[succVtemp[vi]].item() != label.item():
                    self.offspringVs[succVtemp[vi]].Rmax = ADB
                    self.chosen_v = succVtemp[vi]
                    if self.offspringVs[succVtemp[vi]].Rmax <= aim_r:
                        self.chosen_v = succVtemp[vi]
                        return query
                    vi = vi + 1
                else:
                    self.offspringVs[succVtemp[vi]].Rmin = ADB
                    succVtemp.pop(vi)

            if len(succVtemp) == 0:
                low = ADB
            elif len(succVtemp) == 1:
                self.chosen_v = succVtemp[0]
                return query
            elif len(succVtemp) >= 2:
                high = ADB
                succV = succVtemp
        
        # d1 and d2 are close, just return d1
        self.chosen_v = succV[0]
        return query


# ==================== Progress Display ====================

def progress_bar(imgi, query, iter, Rnow):
    """Show the results dynamically."""
    sys.stdout.write(f'\rImg{imgi} Query{query:.0f}\t Iter{iter:.0f}\t Rinf{Rnow:.4f}')
    sys.stdout.flush()


# ==================== Algorithm 1: Main Attack Function ====================

def ATK_ADBA(model, device, original_image_x, img_number, label_y, aim_r, 
             tolerance_binary_iters, initDir, offspringN, binaryM, budget, channels=None):

    model.eval()
    
    # Extract dimensions
    _, size_channel, size_x, size_y = original_image_x.shape
    if channels is not None and channels == 1:
        size_channel = channels
    pix_num = size_channel * size_x * size_y
    
    # Initialize perturbation direction
    v0 = V(size_channel, size_x, size_y, initDir, device)

    iter_num = 1
    block_iter = 0
    b0 = Block(0, pix_num - 1)
    bs1 = b0.cut_block(offspringN)
    blocks = [bs1]

    query = 0
    ITERATION = Iter(v0, offspringN, device, 1)
    
    # First mutation
    query = query + ITERATION.mutation(
        model, original_image_x, label_y, aim_r, tolerance_binary_iters, blocks[0], binaryM)
    progress_bar(img_number, query, iter_num, ITERATION.old_vbest.Rmax)

    # Main ATK loop
    while (query < budget) and (ITERATION.old_vbest.Rmax > aim_r):
        block_iter = block_iter + 1
        blocks_i = []
        for i, bi in enumerate(blocks[block_iter - 1]):
            blocks_i.extend(bi.cut_block(offspringN))
            query_plus = ITERATION.mutation(
                model, original_image_x, label_y, aim_r, tolerance_binary_iters,
                blocks_i[offspringN * i:offspringN * (i + 1)], binaryM)
            query = query + query_plus
            progress_bar(img_number, query, iter_num, ITERATION.old_vbest.Rmax)
            iter_num = iter_num + 1
            if (ITERATION.old_vbest.Rmax <= aim_r) or query >= budget:
                break
        blocks.append(copy.deepcopy(blocks_i))

    print()  # New line after progress bar
    
    # Generate final adversarial image
    Rbest = ITERATION.old_vbest.Rmax
    adversarial_v = ITERATION.old_vbest.advv_to_tensor()   
    adversarial_image = original_image_x + Rbest * adversarial_v
    adversarial_image = torch.clamp(adversarial_image, 0.0, 1.0)

    success = 1 if Rbest <= aim_r else -1

    return success, query, ITERATION.iter_n, Rbest, adversarial_image, adversarial_v    # added adversarial_v for vbest return


# ==================== Wrapper Function for adba_attack.py ====================

def ADBA_Attack(model, device, original_image, label, epsilon, budget, init_dir=1, offspring_n=2, binary_mode=1, channels=None):
    
    # Check if model correctly classifies original image
    model.eval()
    with torch.no_grad():
        pred = model(original_image).argmax(1)
    
    if pred.item() != label.item():
        # Model already misclassifies - return original image
        return original_image, True, 0, 0.0
    
    # Run ATK_ADBA
    success_int, queries, iter_n, Rbest, adv_image, Vbest = ATK_ADBA(    # return vbest to use to find noisy image at epsilon in best v direction
        model=model, 
        device=device,
        original_image_x=original_image,
        img_number=0,  # Single image, use 0
        label_y=label,
        aim_r=epsilon,
        tolerance_binary_iters=8,  # Default from original
        initDir=init_dir,
        offspringN=offspring_n,
        binaryM=binary_mode,
        budget=budget,
        channels=channels
    )
    
    # Convert success from int (1/-1) to bool
    success = (success_int == 1)
       
    if success:
        return adv_image, True, queries, Rbest
    else:
        # Attack failed: return image with perturbation at exactly epsilon using the best direction found (Vbest)
        epsilon_bounded_adv = torch.clamp(original_image + epsilon * Vbest, 0.0, 1.0)   
        return epsilon_bounded_adv, False, queries, epsilon