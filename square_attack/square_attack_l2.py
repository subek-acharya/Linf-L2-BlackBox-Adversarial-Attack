import numpy as np
import torch
from . import square_attack_utils
import utils
import time

def SquareAttackL2_Wrapper(model, device, dataLoader, eps=0.5, n_iters=1000, p_init=0.1, n_classes=2, targeted=False):
    # Collect all original examples and labels
    x_tensor, y_tensor = utils.DataLoaderToTensor(dataLoader)
    print("x_tensor: ", x_tensor.shape)
    all_original_examples, all_labels = utils.TensorToNumpy(x_tensor, y_tensor)
    
    # Convert labels to one-hot
    y_onehot = square_attack_utils.dense_to_onehot(all_labels, n_classes)

    # All samples are correctly classified (already filtered in main.py)
    corr_classified = np.ones(len(all_labels), dtype=bool)
    
    # Run attack
    n_queries, all_adv_images = square_attack_l2(
        model=model,
        device=device,
        x=all_original_examples,
        y=y_onehot,
        corr_classified=corr_classified,
        eps=eps,
        n_iters=n_iters,
        p_init=p_init,
        targeted=targeted,
        loss_type="margin_loss",
    )

    # Convert numpy arrays back to tensors using NumpyToTensor
    xAdv, yClean = utils.NumpyToTensor(all_adv_images, all_labels)

    print("n_queries: ", n_queries.shape)
    
    # Create and return adversarial dataLoader
    advLoader = utils.TensorToDataLoader(xAdv, yClean, transforms=None, batchSize=dataLoader.batch_size, randomizer=None)
    
    return advLoader


def square_attack_l2(model, device, x, y, corr_classified, eps, n_iters, p_init, targeted, loss_type):
    """ The L2 square attack """
    np.random.seed(0)

    min_val, max_val = 0, 1
    c, h, w = x.shape[1:]
    n_features = c * h * w
    n_ex_total = x.shape[0]
    x, y = x[corr_classified], y[corr_classified]

    ### initialization with pseudo-gaussian perturbations
    delta_init = np.zeros(x.shape)
    s = h // 5
    sp_init = (h - s * 5) // 2
    center_h = sp_init + 0
    for counter in range(h // s):
        center_w = sp_init + 0
        for counter2 in range(w // s):
            delta_init[:, :, center_h:center_h + s, center_w:center_w + s] += square_attack_utils.meta_pseudo_gaussian_pert(s).reshape(
                [1, 1, s, s]) * np.random.choice([-1, 1], size=[x.shape[0], c, 1, 1])
            center_w += s
        center_h += s

    # Normalize and scale by epsilon
    # x_best = np.clip(x + delta_init / np.sqrt(np.sum(delta_init ** 2, axis=(1, 2, 3), keepdims=True) + 1e-10) * eps, min_val, max_val)
    x_best = np.clip(square_attack_utils.quantize_numpy(x + delta_init / np.sqrt(np.sum(delta_init ** 2, axis=(1, 2, 3), keepdims=True) + 1e-10) * eps), min_val, max_val) # quantization applied

    logits = square_attack_utils.predict(model, x_best, device, batch_size=64)
    loss_min = square_attack_utils.loss(y, logits, targeted, loss_type=loss_type)
    margin_min = square_attack_utils.loss(y, logits, targeted, loss_type='margin_loss')
    n_queries = np.ones(x.shape[0])  # ones because we have already used 1 query

    time_start = time.time()
    s_init = int(np.sqrt(p_init * n_features / c))
    
    for i_iter in range(n_iters):
        idx_to_fool = (margin_min > 0.0)

        x_curr, x_best_curr = x[idx_to_fool], x_best[idx_to_fool]
        y_curr, margin_min_curr = y[idx_to_fool], margin_min[idx_to_fool]
        loss_min_curr = loss_min[idx_to_fool]
        delta_curr = x_best_curr - x_curr

        p = square_attack_utils.p_selection(p_init, i_iter, n_iters)
        s = max(int(round(np.sqrt(p * n_features / c))), 3)

        if s % 2 == 0:
            s += 1

        s2 = s + 0
        
        ### window_1
        center_h = np.random.randint(0, h - s)
        center_w = np.random.randint(0, w - s)
        new_deltas_mask = np.zeros(x_curr.shape)
        new_deltas_mask[:, :, center_h:center_h + s, center_w:center_w + s] = 1.0

        ### window_2
        center_h_2 = np.random.randint(0, h - s2)
        center_w_2 = np.random.randint(0, w - s2)
        new_deltas_mask_2 = np.zeros(x_curr.shape)
        new_deltas_mask_2[:, :, center_h_2:center_h_2 + s2, center_w_2:center_w_2 + s2] = 1.0
        
        ### compute total norm available
        curr_norms_window = np.sqrt(
            np.sum(((x_best_curr - x_curr) * new_deltas_mask) ** 2, axis=(2, 3), keepdims=True) + 1e-10)
        curr_norms_image = np.sqrt(np.sum((x_best_curr - x_curr) ** 2, axis=(1, 2, 3), keepdims=True) + 1e-10)
        mask_2 = np.maximum(new_deltas_mask, new_deltas_mask_2)
        norms_windows = np.sqrt(np.sum((delta_curr * mask_2) ** 2, axis=(2, 3), keepdims=True) + 1e-10)

        ### create the updates
        new_deltas = np.ones([x_curr.shape[0], c, s, s])
        new_deltas = new_deltas * square_attack_utils.meta_pseudo_gaussian_pert(s).reshape([1, 1, s, s])
        new_deltas *= np.random.choice([-1, 1], size=[x_curr.shape[0], c, 1, 1])
        old_deltas = delta_curr[:, :, center_h:center_h + s, center_w:center_w + s] / (1e-10 + curr_norms_window)
        new_deltas += old_deltas
        new_deltas = new_deltas / np.sqrt(np.sum(new_deltas ** 2, axis=(2, 3), keepdims=True) + 1e-10) * (
            np.maximum(eps ** 2 - curr_norms_image ** 2, 0) / c + norms_windows ** 2) ** 0.5
        
        delta_curr[:, :, center_h_2:center_h_2 + s2, center_w_2:center_w_2 + s2] = 0.0  # set window_2 to 0
        delta_curr[:, :, center_h:center_h + s, center_w:center_w + s] = new_deltas + 0  # update window_1

        x_new = x_curr + delta_curr / np.sqrt(np.sum(delta_curr ** 2, axis=(1, 2, 3), keepdims=True) + 1e-10) * eps
        # x_new = np.clip(x_new, min_val, max_val)
        x_new = np.clip(square_attack_utils.quantize_numpy(x_new), min_val, max_val) # quantization applied
        
        curr_norms_image = np.sqrt(np.sum((x_new - x_curr) ** 2, axis=(1, 2, 3), keepdims=True))

        logits = square_attack_utils.predict(model, x_new, device, batch_size=64)
        loss = square_attack_utils.loss(y_curr, logits, targeted, loss_type=loss_type)
        margin = square_attack_utils.loss(y_curr, logits, targeted, loss_type='margin_loss')

        idx_improved = loss < loss_min_curr
        loss_min[idx_to_fool] = idx_improved * loss + ~idx_improved * loss_min_curr
        margin_min[idx_to_fool] = idx_improved * margin + ~idx_improved * margin_min_curr

        idx_improved = np.reshape(idx_improved, [-1, *[1] * len(x.shape[:-1])])
        x_best[idx_to_fool] = idx_improved * x_new + ~idx_improved * x_best_curr
        n_queries[idx_to_fool] += 1

        acc = (margin_min > 0.0).sum() / n_ex_total
        acc_corr = (margin_min > 0.0).mean()
        mean_nq = np.mean(n_queries)
        mean_nq_ae = np.mean(n_queries[margin_min <= 0]) if np.any(margin_min <= 0) else np.mean(n_queries)
        median_nq = np.median(n_queries)
        median_nq_ae = np.median(n_queries[margin_min <= 0]) if np.any(margin_min <= 0) else np.median(n_queries)
        
        time_total = time.time() - time_start
        print('{}: acc={:.2%} acc_corr={:.2%} avg#q_ae={:.1f} med#q_ae={:.1f} s={}->{}, n_ex={}, {:.0f}s, loss={:.3f}, max_pert={:.1f}, impr={:.0f}'.
            format(i_iter + 1, acc, acc_corr, mean_nq_ae, median_nq_ae, s_init, s, x.shape[0], time_total,
                   np.mean(margin_min), np.amax(curr_norms_image), np.sum(idx_improved)))

        if acc == 0:
            curr_norms_image = np.sqrt(np.sum((x_best - x) ** 2, axis=(1, 2, 3), keepdims=True))
            print('Maximal norm of the perturbations: {:.5f}'.format(np.amax(curr_norms_image)))
            break

    curr_norms_image = np.sqrt(np.sum((x_best - x) ** 2, axis=(1, 2, 3), keepdims=True))
    print('Maximal norm of the perturbations: {:.5f}'.format(np.amax(curr_norms_image)))

    return n_queries, x_best