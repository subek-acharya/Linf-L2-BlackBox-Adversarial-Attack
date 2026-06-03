import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from constants import EXPERIMENTS_UNET_ALL, UNET_CHECKPOINT
from rays_attack import RaySAttack
import utils
from ModelFactory import ModelFactory


class RaysAttackExperiment:
    def __init__(
        self, experiments_config, epsilon_max, query_limit=10000, total_samples=500, n_classes=2,
    ):
        self.experiments = experiments_config
        self.epsilon_max = epsilon_max
        self.query_limit = query_limit
        self.total_samples = total_samples
        self.n_classes = n_classes
        self.results = {}
        self.headers_written = set()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.factory = ModelFactory(device=self.device)

    def run_all(self, batch_size=64):
        for model_name, config in self.experiments.items():
            use_unet = config.get("use_unet", False)
            unet_ckpt = config.get("unet_ckpt", UNET_CHECKPOINT) if use_unet else None
            mode_str = "UNet + Model" if use_unet else "Model Only"

            print(
                f"Model: {model_name} ({mode_str}), Dataset: {config['dataset_path']}, Epsilon Max: {self.epsilon_max}, Query Limit: {self.query_limit}, Total Samples: {self.total_samples}"
            )

            model = self._load_model(model_name, config, use_unet, unet_ckpt)

            loader = self._get_clean_loader(config["dataset_path"], model, batch_size)
            if not loader:
                continue

            adv_loader, acc = self._execute_attack(model, loader)
            
            unet_suffix = "_unet" if use_unet else ""
            result_key = f"{model_name}{unet_suffix}_eps={int(self.epsilon_max * 255)}/255"
            self.results[result_key] = acc

            self._append_result(result_key, acc, use_unet)

            if adv_loader:
                self._save_samples(
                    adv_loader, model, model_name, config["dataset_path"], use_unet
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
            adv_loader = RaySAttack(
                self.device, model, self.epsilon_max, self.query_limit, loader
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

    def _save_samples(self, adv_loader, model, model_name, dataset_path, use_unet, n_save=500):
        unet_suffix = "_unet" if use_unet else ""
        base_dir = os.path.join("adv_samples", f"rays{unet_suffix}", model_name)
        os.makedirs(base_dir, exist_ok=True)

        clean_name = (
            os.path.basename(dataset_path).replace(".pth", "").replace(".pt", "").replace("val_", "")
        )
        filename = (
            f"{model_name}{unet_suffix}_eps={int(self.epsilon_max * 255)}by255_{clean_name}.pt"
        )
        save_path = os.path.join(base_dir, filename)

        x_adv, y_clean = utils.DataLoaderToTensor(adv_loader)
        x_adv = x_adv[:n_save]
        y_clean = y_clean[:n_save]

        output_data = {
            "data": x_adv.float(),
            "binary_labels": y_clean.long(),
            "original_labels": y_clean.long(),
        }
        
        torch.save(output_data, save_path)
        print(f"Saved {n_save} samples to {save_path}")

    def _append_result(self, result_key, acc, use_unet, filepath=None):
        if filepath is None:
            filepath = "rays_unet_results.txt" if use_unet else "rays_results.txt"
        
        eps_key = int(self.epsilon_max * 255)
        mode_str = "(UNet+Model)" if use_unet else ""

        with open(filepath, "a", encoding="utf-8") as f:
            header_key = (eps_key, use_unet)
            if header_key not in self.headers_written:
                f.write(
                    "\n"
                    + "=" * 10
                    + f" RAYS ATTACK {mode_str} FINAL RESULTS eps={eps_key}/255 "
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
        experiment = RaysAttackExperiment(
            experiments_config=EXPERIMENTS_UNET_ALL,
            epsilon_max=eps,
        )
        experiment.run_all()


if __name__ == "__main__":
    main()