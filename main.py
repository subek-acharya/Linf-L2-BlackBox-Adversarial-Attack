import torch
from torch.utils import data
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from model_architecture import ResNet, cait, VGG
import linf_attack
import l2_attack
import utils

def main():

    # --------- ResNet models ---------
    modelDir="./checkpoint/ModelResNet20-VotingOnlyBubbles-v2-Grayscale-Run1.th"
    # modelDir = "./checkpoint/ModelResNet20-VotingCombined-v2-Grayscale-Run1.th"
    # --------- CaiT models ----------
    # modelDir = "./checkpoint/ModelCaiT-VotingOnlyBubbles-v2-Grayscale-Run1.th"
    # modelDir = "./checkpoint/ModelCaiT-trCombined-v2-valCombined-v2-Grayscale-Run1.th"
    # --------- VGG models -----------
    # modelDir = "./checkpoint/ModelVgg16-B.th"
    # modelDir = "./checkpoint/ModelVgg16-C2.th"

    # Parameters for the dataset
    batchSize = 64
    numClasses = 2
    inputImageSize = [1, 1, 40, 50]
    dropOutRate = 0.0

    # Define GPU device
    device = torch.device("cuda")

    # -------------- Loading ResNet model ------------------
    # Create the ResNet model
    model = ResNet.resnet20(inputImageSize, dropOutRate, numClasses).to(device)
    # Load in the trained weights of the model
    checkpoint = torch.load(modelDir, weights_only=False)
    # Load the state dictionary into the model
    model.load_state_dict(checkpoint["state_dict"])

    # -------------- Loading CaiT model ---------------------
    # model = cait.CaiT(
    #     image_size   =(40, 50),
    #     patch_size   =5,
    #     num_classes  =2,
    #     num_channels =1,
    #     dim          =512,
    #     depth        =16,
    #     cls_depth    =2,
    #     heads        =8,
    #     mlp_dim      =2048,
    #     dropout      =0.1,
    #     emb_dropout  =0.1,
    #     layer_dropout=0.05
    # ).to(device)

    # ckpt = torch.load(modelDir, map_location=device, weights_only=False)
    # model.load_state_dict(ckpt["state_dict"])
    
    # # IMPORTANT: CaiT drops layers even in eval(); disable for deterministic eval/attack
    # model.patch_transformer.layer_dropout = 0.0
    # model.cls_transformer.layer_dropout   = 0.0

    # ----------------- Loading VGG model ----------------------
    # imgH, imgW   = 40, 50
    # # Create the VGG16 model
    # model = VGG.VGG("VGG16", imgH, imgW, numClasses).to(device)
    # # Load checkpoint (handles raw state_dict or dict with 'state_dict'; strips 'module.' if present)
    # raw = torch.load(modelDir, map_location="cpu")
    # state = raw.get("state_dict", raw)
    # state = { (k[7:] if k.startswith("module.") else k): v for k, v in state.items() }
    
    # missing, unexpected = model.load_state_dict(state, strict=False)
    # print(f"Loaded checkpoint (strict=False) | missing={len(missing)} unexpected={len(unexpected)}")

    # ----------------------------------------------------------

    # Switch the model into eval model for testing
    model= model.eval()
    
    # ---------------- dataset --------------
    # data = torch.load("./data/kaleel_final_dataset_train_OnlyBubbles_Grayscale.pth", weights_only=False)
    data = torch.load("./data/kaleel_final_dataset_val_OnlyBubbles_Grayscale.pth", weights_only=False)
    # data = torch.load("./data/kaleel_final_dataset_train_Combined_Grayscale.pth", weights_only=False)
    # data = torch.load("./data/kaleel_final_dataset_val_Combined_Grayscale.pth", weights_only=False)
    images = data["data"].float()
    labels_binary = data["binary_labels"].long()
    labels_original = data["original_labels"].long()

    print(f"Image shape: {images.shape}")

    # Create a dataloader with only images and binary labels
    dataset = TensorDataset(images, labels_binary)
    valLoader = DataLoader(dataset, batch_size = batchSize, shuffle = False)

    # Check the clean accuracy of the model
    cleanAcc = utils.validateD(valLoader, model, device)
    print("Voter Dataset Val Loader Clean Acc:", cleanAcc)

    # Get correctly classified, classwise balanced samples to do the attack
    totalSamplesRequired = 100
    correctLoader = utils.GetCorrectlyIdentifiedSamplesBalanced(model, totalSamplesRequired, valLoader, numClasses)

    # Check the number of samples in the correctLoader
    print("Number of samples in correctLoader:", len(correctLoader.dataset))

    # Get pixel bounds for a correctLoader
    minVal, maxVal = utils.GetDataBounds(correctLoader, device)
    print("Data Range for Correct Loader:", [round(minVal, 4), round(maxVal, 4)])

    # RayS Attack Parameters
    # epsilonMax = 1
    # queryLimit = 1000

    #-------------- Linf Attack - RaySAttack ------------
    # advLoaderRayS = linf_attack.RaySAttack(model, epsilonMax, queryLimit, correctLoader)
    # advAcc = utils.validateD(advLoaderRayS, model, device)
    # print("RayS black-box robustness:", advAcc)

    #----------------------- L2 Attack - Surfee Attack ---------------
    
    # Configuration for SurFree attack
    surfree_config = {
        "init": {
            "steps": 100,                       # Number of optimization steps
            "max_queries": 500,                # Maximum queries per image
            "BS_gamma": 0.01,                   # Binary search precision threshold
            "BS_max_iteration": 7,              # Max binary search iterations
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
                "basis_type": "random",            # Type of basis ("dct" or "random")
                "dct_type": "full",             # DCT type ("full" or "8x8")
                "frequence_range": (0, 0.5),    # Frequency range for DCT
                "beta": 0.001,                  # Noise factor for DCT basis
                "tanh_gamma": 1,                # Gamma for tanh function in DCT
                "random_noise": "normal"        # Noise type ("normal" or "uniform")
            }
        }
    }
    
    # # Run SurFree attack
    advLoader = l2_attack.SurFree_AttackWrapper(
        model=model,
        device=device,
        dataLoader=correctLoader,
        config=surfree_config
    )
    
    # # Calculate adversarial accuracy
    advAcc = utils.validateD(advLoader, model, device)
    print("\n" + "=" * 60)
    print(f"SurFree Attack Results:")
    print(f"  - Adversarial Accuracy: {advAcc}")

if __name__ == '__main__':
    main()