import os
import math
import torch
from torch.utils.data import DataLoader, TensorDataset
from constants import EXPERIMENTS
from rays_attack import RaySAttack
import utils
from ModelFactory import ModelFactory


class RaysAttackExperiment:
    def __init__(
        self, experiments_config, epsilon_max, query_limit=10000, total_samples=1000
    ):
        self.experiments = experiments_config
        self.epsilon_max = epsilon_max
        self.query_limit = query_limit
        self.total_samples = total_samples
        self.results = {}
        self.headers_written = set()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run_all(self, batch_size=64):
        for model_name, config in self.experiments.items():
            print(
                f"Model: {model_name}, Dataset: {config['dataset_path']}, Epsilon Max: {self.epsilon_max}, Query Limit: {self.query_limit}, Total Samples: {self.total_samples}"
            )

            model = ModelFactory().get_model(model_name, config["ckpt_path"])
            model.to(self.device)

            loader = self._get_clean_loader(config["dataset_path"], model, batch_size)
            if not loader:
                continue

            adv_loader, acc = self._execute_attack(model, loader)
            self.results[model_name + f"_eps={int(self.epsilon_max * 255)}/255"] = acc

            self._append_result(
                model_name + f"_eps={int(self.epsilon_max * 255)}/255", acc
            )

            if adv_loader:
                self._save_samples(
                    adv_loader, model, model_name, config["dataset_path"]
                )

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
            return None, None

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
            return None

    def _save_samples(self, adv_loader, model, model_name, dataset_path, n_save=20):
        base_dir = os.path.join("adv_samples", "rays", model.__class__.__name__)
        os.makedirs(base_dir, exist_ok=True)

        clean_name = (
            os.path.basename(dataset_path).replace(".pth", "").replace("val_", "")
        )
        filename = (
            f"{model_name}_eps={int(self.epsilon_max * 255)}by255_{clean_name}.pt"
        )
        save_path = os.path.join(base_dir, filename)

        imgs, lbls = [], []
        count = 0
        for x, y in adv_loader:
            imgs.append(x.cpu())
            lbls.append(y.cpu())
            count += x.size(0)
            if count >= n_save:
                break

        final_imgs = torch.cat(imgs)[:n_save]
        final_lbls = torch.cat(lbls)[:n_save]
        torch.save({"images": final_imgs, "labels": final_lbls}, save_path)
        print(f"Saved {n_save} samples to {save_path}")

    def _append_result(self, model_name, acc, filepath="results_new_env.txt"):
        eps_key = int(self.epsilon_max * 255)

        with open(filepath, "a", encoding="utf-8") as f:
            if eps_key not in self.headers_written:
                f.write(
                    "\n"
                    + "=" * 10
                    + f" RAYS ATTACK FINAL RESULTS eps={eps_key}/255"
                    + "=" * 10
                    + "\n"
                )
                self.headers_written.add(eps_key)

            if acc is not None:
                f.write(f"{model_name:<30}: {acc:.4f}\n")
            else:
                f.write(f"{model_name:<30}: Failed\n")


def main():
    epsilon = [255 / 255, 8 / 255, 4 / 255]
    # epsilon = [255 / 255]
    for eps in epsilon:
        experiment = RaysAttackExperiment(
            experiments_config=EXPERIMENTS,
            epsilon_max=eps,
        )
        experiment.run_all()


if __name__ == "__main__":
    main()
