import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from constants import EXPERIMENTS
from surfree_attack import SurFree_AttackWrapper
import utils
from ModelFactory import ModelFactory


class SurfreeAttackExperiment:
    def __init__(
        self, experiments_config, surfree_config, total_samples=1000
    ):
        self.experiments = experiments_config
        self.surfree_config = surfree_config
        self.total_samples = total_samples
        self.results = {}
        self.headers_written = set()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # L2 distance thresholds for saving
        self.thresholds = [1, 2, 3, 5, 15, 45]

    def run_all(self, batch_size=64):
        for model_name, config in self.experiments.items():
            print(
                f"Model: {model_name}, Dataset: {config['dataset_path']}, Max Queries: {self.surfree_config['init']['max_queries']}, Steps: {self.surfree_config['init']['steps']}, Total Samples: {self.total_samples}"
            )

            model = ModelFactory().get_model(model_name, config["ckpt_path"])
            model.to(self.device)
            model.eval()

            loader = self._get_clean_loader(config["dataset_path"], model, batch_size)
            if not loader:
                continue

            # Get clean accuracy before attack
            clean_acc = utils.validateD(loader, model, self.device)
            print(f"Clean Accuracy: {clean_acc:.4f}")

            adv_loader, adv_blobs, acc = self._execute_attack(model, loader)
            
            # Store results
            result_key = f"{model_name}_SurFree_L2"
            self.results[result_key] = {
                "clean_acc": clean_acc,
                "robust_acc": acc,
                "adv_blobs": adv_blobs
            }

            self._append_result(model_name, clean_acc, acc, adv_blobs)

            if adv_blobs:
                self._save_samples(
                    adv_blobs, model, model_name, config["dataset_path"]
                )

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
            dataset = TensorDataset(data["data"].float(), data["binary_labels"].long())
            raw_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

            return utils.GetCorrectlyIdentifiedSamplesBalanced(
                model, self.total_samples, raw_loader, numClasses=2
            )
        except Exception as e:
            print(f"Dataloader failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _save_samples(self, adv_blobs, model, model_name, dataset_path):
        base_dir = os.path.join("adv_samples", "surfree", model.__class__.__name__)
        os.makedirs(base_dir, exist_ok=True)
    
        clean_name = (
            os.path.basename(dataset_path).replace(".pth", "").replace("val_", "")
        )
        
        for threshold, blob in adv_blobs.items():
            n_samples = blob["x_clean"].shape[0]
            
            if n_samples == 0:
                print(f"No samples for L2 <= {threshold}, skipping save.")
                continue
    
            filename = f"{model_name}_L2_leq_{threshold}_{clean_name}.pth"
            save_path = os.path.join(base_dir, filename)
            
            torch.save(blob, save_path)
            
            print(f"Saved {n_samples} samples to {save_path}")

    def _append_result(self, model_name, clean_acc, robust_acc, adv_blobs, filepath="results_surfree.txt"):
        
        with open(filepath, "a", encoding="utf-8") as f:
            # Write header if not written yet
            if "surfree" not in self.headers_written:
                f.write(
                    "\n"
                    + "=" * 10
                    + " SURFREE L2 ATTACK RESULTS "
                    + "=" * 10
                    + "\n"
                )
                self.headers_written.add("surfree")

            f.write(f"\n--- {model_name} ---\n")
            
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
    # L2 distance thresholds  = [1, 2, 3, 5, 15, 45]
    experiment = SurfreeAttackExperiment(
        experiments_config=EXPERIMENTS,
        surfree_config=DEFAULT_SURFREE_CONFIG,
        total_samples=1000
    )
    experiment.run_all(batch_size=64)


if __name__ == "__main__":
    main()