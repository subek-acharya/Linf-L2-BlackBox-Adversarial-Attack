import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from constants import EXPERIMENTS_UNET_ALL, UNET_CHECKPOINT
from surfree_attack import SurFree_AttackWrapper
import utils
from ModelFactory import ModelFactory

class SurfreeAttackExperiment:
    def __init__(
        self, experiments_config, surfree_config, total_samples=500, n_classes=2,
    ):
        self.experiments = experiments_config
        self.surfree_config = surfree_config
        self.total_samples = total_samples
        self.n_classes = n_classes
        self.results = {}
        self.headers_written = set()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.factory = ModelFactory(device=self.device)
        
        # L2 distance thresholds for saving
        self.thresholds = [1, 2, 3, 5, 15, 45]

    def run_all(self, batch_size=64):
        for model_name, config in self.experiments.items():
            use_unet = config.get("use_unet", False)
            unet_ckpt = config.get("unet_ckpt", UNET_CHECKPOINT) if use_unet else None
            mode_str = "UNet + Model" if use_unet else "Model Only"

            print(
                f"Model: {model_name} ({mode_str}), Dataset: {config['dataset_path']}, "
                f"Max Queries: {self.surfree_config['init']['max_queries']}, "
                f"Steps: {self.surfree_config['init']['steps']}, Total Samples: {self.total_samples}"
            )

            model = self._load_model(model_name, config, use_unet, unet_ckpt)

            loader = self._get_clean_loader(config["dataset_path"], model, batch_size)
            if not loader:
                continue

            # Get clean accuracy before attack
            clean_acc = utils.validateD(loader, model, self.device)
            print(f"Clean Accuracy: {clean_acc:.4f}")

            adv_loader, adv_blobs, acc = self._execute_attack(model, loader)
            
            # Store results
            unet_suffix = "_unet" if use_unet else ""
            result_key = f"{model_name}{unet_suffix}_SurFree_L2"
            self.results[result_key] = {
                "clean_acc": clean_acc,
                "robust_acc": acc,
                "adv_blobs": adv_blobs
            }

            self._append_result(model_name, clean_acc, acc, adv_blobs, use_unet)

            if adv_blobs:
                self._save_samples(
                    adv_blobs, model_name, config["dataset_path"], use_unet
                )

    def _load_model(self, model_name, config, use_unet, unet_ckpt):
        """Load model with or without UNet wrapper."""
        if use_unet:
            model = self.factory.get_unet_model_wrapper(
                model_name=model_name,
                model_checkpoint=config["ckpt_path"],
                unet_checkpoint=unet_ckpt,
            )
        else:
            model = self.factory.get_model(model_name, config["ckpt_path"])
        
        model.to(self.device)
        model.eval()
        return model

    def _execute_attack(self, model, loader):
        try:
            adv_loader, adv_blobs = SurFree_AttackWrapper(
                model=model,
                device=self.device,
                dataLoader=loader,
                config=self.surfree_config
            )
            
            acc = utils.validateD(adv_loader, model, self.device)
            print(f"\nOverall Robust Accuracy: {acc:.8f}")
            
            return adv_loader, adv_blobs, acc
            
        except Exception as e:
            print(f"Attack failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None

    def _get_clean_loader(self, dataset_path, model, batch_size):
        try:
            data = torch.load(dataset_path, weights_only=False)
            
            # Handle different data formats (scanned vs original)
            if "xData" in data:
                images = data["xData"].float()
                labels = data["yDataBinary"].long()
            elif "data" in data:
                images = data["data"].float()
                labels = data["binary_labels"].long()
            else:
                raise ValueError(f"Unknown data format. Keys: {list(data.keys())}")
            
            dataset = TensorDataset(images, labels)
            raw_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

            return utils.GetCorrectlyIdentifiedSamplesBalanced(
                model, self.total_samples, raw_loader, numClasses=self.n_classes
            )
        except Exception as e:
            print(f"Dataloader failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _convert_surfree_to_target_format(self, blob):
        """
        Convert SurFree output format to target format.
        
        Original format (SurFree):
            "x_clean":      x_clean         # [N, 1, 40, 50], float32
            "x_adv":        x_adv           # [N, 1, 40, 50], float32
            "labels":       labels          # [N], int64
            "l2_distances": l2_distances    # [N], float32
            "threshold":    threshold       # int
        
        Target format:
            "data":            x_adv              # [N, 1, 40, 50], float32
            "binary_labels":   labels             # [N], int64 (0/1)
            "original_labels": labels             # same as binary_labels
        """
        x_adv = blob["x_adv"]
        labels = blob["labels"]
        
        target_data = {
            "data": x_adv.float(),
            "binary_labels": labels.long(),
            "original_labels": labels.long(),
        }
        
        return target_data

    def _save_samples(self, adv_blobs, model_name, dataset_path, use_unet):
        unet_suffix = "_unet" if use_unet else ""
        base_dir = os.path.join("adv_samples", f"surfree{unet_suffix}", model_name)
        os.makedirs(base_dir, exist_ok=True)
    
        clean_name = (
            os.path.basename(dataset_path).replace(".pth", "").replace(".pt", "").replace("val_", "")
        )
        
        for threshold, blob in adv_blobs.items():
            n_samples = blob["x_clean"].shape[0]
            
            if n_samples == 0:
                print(f"No samples for L2 <= {threshold}, skipping save.")
                continue
            
            # Convert to target format
            target_data = self._convert_surfree_to_target_format(blob)
    
            filename = f"{model_name}{unet_suffix}_L2_leq_{threshold}_{clean_name}.pt"
            save_path = os.path.join(base_dir, filename)
            
            torch.save(target_data, save_path)
            
            print(f"\n{'─' * 60}")
            print(f"Saved {n_samples} samples to {save_path}")
            print(f"  data shape: {target_data['data'].shape}")
            print(f"  binary_labels: {torch.bincount(target_data['binary_labels']).tolist()}")
            print(f"  L2 threshold: {threshold}")
            print(f"{'─' * 60}")

    def _append_result(self, model_name, clean_acc, robust_acc, adv_blobs, use_unet, filepath=None):
        if filepath is None:
            filepath = "surfree_unet_results.txt" if use_unet else "surfree_results.txt"
        
        mode_str = "(UNet+Model)" if use_unet else ""
        unet_suffix = "_unet" if use_unet else ""
        
        with open(filepath, "a", encoding="utf-8") as f:
            # Write header if not written yet
            header_key = ("surfree", use_unet)
            if header_key not in self.headers_written:
                f.write(
                    "\n"
                    + "=" * 10
                    + f" SURFREE L2 ATTACK {mode_str} RESULTS "
                    + "=" * 10
                    + "\n"
                )
                self.headers_written.add(header_key)

            f.write(f"\n--- {model_name}{unet_suffix} ---\n")
            
            if clean_acc is not None:
                f.write(f"  Clean Accuracy: {clean_acc:.4f}\n")
            
            if robust_acc is not None:
                f.write(f"  Overall Robust Accuracy: {robust_acc:.4f}\n")
                
                # Write per-threshold statistics
                if adv_blobs:
                    f.write(f"  Per-Threshold Sample Counts:\n")
                    for threshold in self.thresholds:
                        if threshold in adv_blobs:
                            blob = adv_blobs[threshold]
                            n_samples = blob["x_clean"].shape[0]
                            if n_samples > 0:
                                mean_l2 = blob["l2_distances"].mean().item()
                                f.write(f"    L2 <= {threshold:2d}: {n_samples:4d} samples, Mean L2: {mean_l2:.4f}\n")
                            else:
                                f.write(f"    L2 <= {threshold:2d}:    0 samples\n")
            else:
                f.write(f"  Attack Failed\n")

# ------------------ Default SurFree Configuration ------------------------------------
DEFAULT_SURFREE_CONFIG = {
    "init": {
        "steps": 100,                      # Number of optimization steps
        "max_queries": 10000,               # Maximum queries per image
        "BS_gamma": 0.001,                  # Binary search precision threshold
        "BS_max_iteration": 10,             # Max binary search iterations
        "theta_max": 30,                    # Max angle for direction search (degrees)
        "n_ortho": 100,                     # Number of orthogonal directions to maintain
        "rho": 0.95,                        # Angle adjustment factor
        "T": 1,                             # Number of evaluations per direction
        "with_alpha_line_search": True,     # Enable binary search on theta
        "with_distance_line_search": False, # Enable binary search on distance
        "with_interpolation": False,        # Enable interpolation
        "final_line_search": True,          # Perform final line search
        "quantification": True,             # Quantify to valid pixel values
        "clip": True                        # Clip values to [0, 1]
    },
    "run": {
        "basis_params": {
            "basis_type": "random",         # Type of basis ("dct" or "random")
            "dct_type": "full",             # DCT type ("full" or "8x8")
            "frequence_range": (0, 0.5),    # Frequency range for DCT
            "beta": 0.001,                  # Noise factor for DCT basis
            "tanh_gamma": 1,                # Gamma for tanh function in DCT
            "random_noise": "normal"        # Noise type ("normal" or "uniform")
        }
    }
}

def main():
    # L2 distance thresholds = [1, 2, 3, 5, 15, 45]
    experiment = SurfreeAttackExperiment(
        experiments_config=EXPERIMENTS_UNET_ALL,
        surfree_config=DEFAULT_SURFREE_CONFIG,
        total_samples=500
    )
    experiment.run_all(batch_size=64)

if __name__ == "__main__":
    main()