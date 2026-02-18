import numpy as np
import torch
from . import square_attack_utils
import utils
import time

def SquareAttackLinf_Wrapper(model, device, dataLoader, eps=0.05, n_iters=1000, p_init=0.05, n_classes=2, targeted=False, loss_type="cross_entropy"):
    # Collect all original examples and labels
    x_tensor, y_tensor = utils.DataLoaderToTensor(dataLoader)
    print("x_tensor: ", x_tensor.shape)
    all_original_examples, all_labels = utils.TensorToNumpy(x_tensor, y_tensor)
    
    # Convert true labels to one-hot
    y_onehot = square_attack_utils.dense_to_onehot(all_labels, n_classes)

    # Generate target labels for targeted attack
    if targeted:
        # Create target labels (any class except the true class)
        y_target = np.zeros_like(all_labels)
        for i in range(len(all_labels)):
            # For binary (2 classes): if true label is 0, target is 1; if true is 1, target is 0
            y_target[i] = 1 - all_labels[i]  # Flip the label for binary case
        
        y_target_onehot = square_attack_utils.dense_to_onehot(y_target, n_classes)
        print(f"True labels (first 10): {all_labels[:10]}")
        print(f"Target labels (first 10): {y_target[:10]}")
    else:
        y_target_onehot = y_onehot  # For untargeted, use true labels

    # All samples are correctly classified (already filtered in main.py)
    corr_classified = np.ones(len(all_labels), dtype=bool)

    print(f"Attack mode: {'Targeted' if targeted else 'Untargeted'}, Loss type: {loss_type}")
    
    # Run attack
    n_queries, all_adv_images = square_attack_linf(
        model=model,
        device=device,
        x=all_original_examples,
        y=y_target_onehot,
        corr_classified=corr_classified,
        eps=eps,
        n_iters=n_iters,
        p_init=p_init,
        targeted=targeted,
        loss_type=loss_type,
    )

    # Convert numpy arrays back to tensors using NumpyToTensor
    xAdv, yClean = utils.NumpyToTensor(all_adv_images, all_labels)

    # print("xAdv: ", xAdv.shape)
    print("n_queries: ", n_queries.shape)
    
    # Create and return adversarial dataLoader
    advLoader = utils.TensorToDataLoader(xAdv, yClean, transforms=None, batchSize=dataLoader.batch_size, randomizer=None)
    
    return advLoader


def square_attack_linf(model, device, x, y, corr_classified, eps, n_iters, p_init, targeted, loss_type):
    """ The Linf square attack """
    np.random.seed(0)  # important to leave it here as well
    min_val, max_val = 0, 1 if x.max() <= 1 else 255
    c, h, w = x.shape[1:]
    n_features = c*h*w
    n_ex_total = x.shape[0]
    x, y = x[corr_classified], y[corr_classified]

    # [c, 1, w], i.e. vertical stripes work best for untargeted attacks
    init_delta = np.random.choice([-eps, eps], size=[x.shape[0], c, 1, w])
    # x_best = np.clip(x + init_delta, min_val, max_val)
    x_best = np.clip(square_attack_utils.quantize_numpy(x + init_delta), min_val, max_val) # quantization applied

    logits = square_attack_utils.predict(model, x_best, device, batch_size=64)
    loss_min = square_attack_utils.loss(y, logits, targeted, loss_type=loss_type)
    margin_min = square_attack_utils.loss(y, logits, targeted, loss_type='margin_loss')
    n_queries = np.ones(x.shape[0])  # ones because we have already used 1 query

    time_start = time.time()
    for i_iter in range(n_iters - 1):
        idx_to_fool = margin_min > 0
        x_curr, x_best_curr, y_curr = x[idx_to_fool], x_best[idx_to_fool], y[idx_to_fool]
        loss_min_curr, margin_min_curr = loss_min[idx_to_fool], margin_min[idx_to_fool]
        deltas = x_best_curr - x_curr

        p = square_attack_utils.p_selection(p_init, i_iter, n_iters)
        for i_img in range(x_best_curr.shape[0]):
            s = int(round(np.sqrt(p * n_features / c)))
            s = min(max(s, 1), h-1)  # at least c x 1 x 1 window is taken and at most c x h-1 x h-1
            center_h = np.random.randint(0, h - s)
            center_w = np.random.randint(0, w - s)

            x_curr_window = x_curr[i_img, :, center_h:center_h+s, center_w:center_w+s]
            x_best_curr_window = x_best_curr[i_img, :, center_h:center_h+s, center_w:center_w+s]
            # prevent trying out a delta if it doesn't change x_curr (e.g. an overlapping patch)
            while np.sum(np.abs(np.clip(x_curr_window + deltas[i_img, :, center_h:center_h+s, center_w:center_w+s], min_val, max_val) - x_best_curr_window) < 10**-7) == c*s*s:
                deltas[i_img, :, center_h:center_h+s, center_w:center_w+s] = np.random.choice([-eps, eps], size=[c, 1, 1])

        # x_new = np.clip(x_curr + deltas, min_val, max_val)
        x_new = np.clip(square_attack_utils.quantize_numpy(x_curr + deltas), min_val, max_val) # quantization applied

        logits = square_attack_utils.predict(model, x_new, device, batch_size=64)
        loss = square_attack_utils.loss(y_curr, logits, targeted, loss_type=loss_type)
        margin = square_attack_utils.loss(y_curr, logits, targeted, loss_type='margin_loss')

        idx_improved = loss < loss_min_curr
        loss_min[idx_to_fool] = idx_improved * loss + ~idx_improved * loss_min_curr
        margin_min[idx_to_fool] = idx_improved * margin + ~idx_improved * margin_min_curr
        idx_improved = np.reshape(idx_improved, [-1, *[1]*len(x.shape[:-1])])
        x_best[idx_to_fool] = idx_improved * x_new + ~idx_improved * x_best_curr
        n_queries[idx_to_fool] += 1

        acc = (margin_min > 0.0).sum() / n_ex_total
        acc_corr = (margin_min > 0.0).mean()
        mean_nq, mean_nq_ae, median_nq_ae = np.mean(n_queries), np.mean(n_queries[margin_min <= 0]), np.median(n_queries[margin_min <= 0])
        avg_margin_min = np.mean(margin_min)
        time_total = time.time() - time_start
        print('{}: acc={:.2%} acc_corr={:.2%} avg#q_ae={:.2f} med#q={:.1f}, avg_margin={:.2f} (n_ex={}, eps={:.3f}, {:.2f}s)'.
            format(i_iter+1, acc, acc_corr, mean_nq_ae, median_nq_ae, avg_margin_min, x.shape[0], eps, time_total))

        if acc == 0:
            break

    return n_queries, x_best