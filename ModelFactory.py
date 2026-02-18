import torch
import torch.nn as nn
from typing import Union, List, Tuple, Optional
from model_architecture import ResNet, cait, VGG, MultiOutputSVM


class ModelFactory:
    def __init__(self, device: Optional[torch.device] = None):
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    def get_model(
        self,
        model_name: str,
        checkpoint_path: Union[str, List[str]],
    ) -> nn.Module:
        model_name = model_name.lower()

        if "resnet" in model_name:
            return self._create_resnet(
                checkpoint_path,
            )
        elif "cait" in model_name:
            return self._create_cait(
                checkpoint_path,
            )
        elif "vgg" in model_name:
            return self._create_vgg16(
                checkpoint_path,
            )
        elif "svm" in model_name:
            if isinstance(checkpoint_path, (list, tuple)) and len(checkpoint_path) == 2:
                return self._create_svm(checkpoint_path[0], checkpoint_path[1])
            else:
                raise ValueError(
                    "SVM requires a list/tuple of two paths: [base_path, multi_path]"
                )
        else:
            raise ValueError(f"Model '{model_name}' not recognized.")

    def _create_resnet(
        self,
        checkpoint_path: str,
        input_size=[1, 1, 40, 50],
        num_classes=2,
        dropout=0.0,
    ) -> nn.Module:
        model = ResNet.resnet20(input_size, dropout, num_classes).to(self.device)
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model

    def _create_cait(self, checkpoint_path: str, num_classes=2) -> nn.Module:
        model = cait.CaiT(
            image_size=(40, 50),
            patch_size=5,
            num_classes=num_classes,
            num_channels=1,
            dim=512,
            depth=16,
            cls_depth=2,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1,
            layer_dropout=0.05,
        ).to(self.device)

        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        model.load_state_dict(checkpoint["state_dict"])

        if hasattr(model, "patch_transformer"):
            model.patch_transformer.layer_dropout = 0.0
        if hasattr(model, "cls_transformer"):
            model.cls_transformer.layer_dropout = 0.0

        model.eval()
        return model

    def _create_vgg16(self, checkpoint_path: str, num_classes=2) -> nn.Module:
        model = VGG.VGG("VGG16", 40, 50, num_classes).to(self.device)

        raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = raw.get("state_dict", raw)

        state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}

        model.load_state_dict(state, strict=False)
        model.eval()
        return model

    def _create_svm(self, base_path: str, multi_path: str) -> nn.Module:
        input_dim = 1 * 40 * 50
        base_state = torch.load(base_path, map_location="cpu", weights_only=False)

        model = MultiOutputSVM.MultiOutputSVM(input_dim, base_state).to(self.device)

        multi_state = torch.load(multi_path, map_location="cpu", weights_only=False)
        model.load_state_dict(multi_state)
        model.eval()
        return model
