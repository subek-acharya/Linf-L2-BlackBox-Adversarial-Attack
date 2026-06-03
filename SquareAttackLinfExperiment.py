import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from constants import EXPERIMENTS_ALL
import square_attack
import utils
from ModelFactory import ModelFactory


class SquareAttackLinfExperiment:
    def __init__(
        self,
        experiments_config,
        epsilon_max,
        n_iters=60000,
        p_init=0.8,
        n_classes=2,
        targeted=False,
        loss_type="cross_entropy",
        total_samples=500,
    ):
        self.experiments = experiments_config
        self.epsilon_max = epsilon_max
        self.n_iters = n_iters
        self.p_init = p_init
        self.n_classes = n_classes
        self.targeted = targeted
        self.loss_type = loss_type
        self.total_samples = total_samples
        self.results = {}
        self.headers_written = set()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.factory = ModelFactory(device=self.device)

    def run_all(self, batch_size=64):
        for model_name, config in self.experiments.items():
            # Check if UNet should be used
            use_unet = config.get("use_unet", False)
            unet_ckpt = config.get("unet_ckpt", UNET_CHECKPOINT) if use_unet else None
            
            mode_str = "UNet + Model" if use_unet else "Model Only"
            
            print(f"\n{'=' * 70}")
            print(f"Model: {model_name} ({mode_str})")
            print(f"Dataset: {config['dataset_path']}")
            print(f"Epsilon: {self.epsilon_max}")
            print(f"N_Iters: {self.n_iters}")
            print(f"Total Samples: {self.total_samples}")
            print(f"{'=' * 70}")

            # Load attack model (UNet+Model or Model only)
            attack_model = self._load_model(model_name, config, use_unet, unet_ckpt)

            # Get clean loader using attack_model for sample selection
            loader = self._get_clean_loader(config["dataset_path"], attack_model, batch_size)
            if not loader:
                continue

            # Execute attack on the same model used for sample selection
            adv_loader, acc = self._execute_attack(attack_model, loader)
            
            # Create result key
            unet_suffix = "_unet" if use_unet else ""
            result_key = f"{model_name}{unet_suffix}_eps={int(self.epsilon_max * 255)}/255"
            self.results[result_key] = acc

            # Save results
            self._append_result(result_key, acc, use_unet)

            if adv_loader:
                self._save_samples(
                    adv_loader, model_name, config["dataset_path"], use_unet
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
            adv_loader = square_attack.SquareAttackLinf_Wrapper(
                model=model,
                device=self.device,
                dataLoader=loader,
                eps=self.epsilon_max,
                n_iters=self.n_iters,
                p_init=self.p_init,
                n_classes=self.n_classes,
                targeted=self.targeted,
                loss_type=self.loss_type,
            )
            acc = utils.validateD(adv_loader, model, self.device)
            print(f"Robust Accuracy (eps={self.epsilon_max}): {acc:.8f}")
            return adv_loader, acc
        except Exception as e:
            print(f"Attack failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def _get_clean_loader(self, dataset_path, model, batch_size):
        """
        Get correctly classified samples using the provided model.
        
        When model is UNet+Model wrapper, samples are selected based on
        UNet+Model's predictions (i.e., samples that UNet+Model correctly classifies).
        """
        try:
            data = torch.load(dataset_path, weights_only=False)
            
            # Handle different data formats (scanned vs original)
            if "xData" in data:
                # Scanned bubble format
                images = data["xData"].float()
                labels = data["yDataBinary"].long()
            elif "data" in data:
                # Original format
                images = data["data"].float()
                labels = data["binary_labels"].long()
            else:
                raise ValueError(f"Unknown data format. Keys: {list(data.keys())}")
            
            dataset = TensorDataset(images, labels)
            raw_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

            # Use the provided model (could be UNet+Model or Model only)
            return utils.GetCorrectlyIdentifiedSamplesBalanced(
                model, self.total_samples, raw_loader, numClasses=self.n_classes
            )
        except Exception as e:
            print(f"Dataloader failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _save_samples(self, adv_loader, model_name, dataset_path, use_unet, n_save=500):
        """Save adversarial samples in standardized format."""
        # Create directory based on mode
        unet_suffix = "_unet" if use_unet else ""
        base_dir = os.path.join("adv_samples", f"square_linf{unet_suffix}", model_name)
        os.makedirs(base_dir, exist_ok=True)

        clean_name = (
            os.path.basename(dataset_path).replace(".pth", "").replace(".pt", "").replace("val_", "")
        )
        filename = (
            f"{model_name}{unet_suffix}_eps={int(self.epsilon_max * 255)}by255_{clean_name}.pt"
        )
        save_path = os.path.join(base_dir, filename)

        # Convert dataloader to tensors using utils
        x_adv, y_clean = utils.DataLoaderToTensor(adv_loader)
        
        # Limit to n_save samples
        x_adv = x_adv[:n_save]
        y_clean = y_clean[:n_save]

        # Save in target format
        output_data = {
            "data": x_adv.float(),
            "binary_labels": y_clean.long(),
            "original_labels": y_clean.long(),
        }
        
        torch.save(output_data, save_path)
        
        print(f"\n{'─' * 60}")
        print(f"Saved {len(x_adv)} samples to {save_path}")
        print(f"  data shape: {output_data['data'].shape}")
        print(f"  binary_labels: {torch.bincount(output_data['binary_labels']).tolist()}")
        print(f"{'─' * 60}")

    def _append_result(self, result_key, acc, use_unet, filepath=None):
        """Append result to file."""
        if filepath is None:
            filepath = "square_unet_results.txt" if use_unet else "square_results.txt"
        
        eps_key = int(self.epsilon_max * 255)
        mode_str = "(UNet+Model)" if use_unet else ""

        with open(filepath, "a", encoding="utf-8") as f:
            header_key = (eps_key, use_unet)
            if header_key not in self.headers_written:
                f.write(
                    "\n"
                    + "=" * 10
                    + f" SQUARE ATTACK LINF {mode_str} FINAL RESULTS eps={eps_key}/255 "
                    + "=" * 10
                    + "\n"
                )
                self.headers_written.add(header_key)

            if acc is not None:
                f.write(f"{result_key:<40}: {acc:.4f}\n")
            else:
                f.write(f"{result_key:<40}: Failed\n")

def main():
    epsilon = [255 / 255, 16 / 255, 8 / 255, 4 / 255]
    # epsilon = [4 / 255]
    
    for eps in epsilon:
        experiment = SquareAttackLinfExperiment(
            experiments_config=EXPERIMENTS_ALL,
            epsilon_max=eps,
        )
        experiment.run_all()


if __name__ == "__main__":
    main()