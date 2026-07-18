from GradCamXai import GradCamXai
from utils import data_transforms, tensor2numpy, NUM_CLASSES
import numpy as np
import torch
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
from torch.utils.data import Subset

XAI_METHODS = {
    'GradCAM': GradCamXai,
    # 'IntegratedGradients': IGXai,
    # 'LIME': LimeXai
}

def produce_explanations(subset_idx_file="correct_preds_indices.npy", model_ckpt="ckpts/rn18_cifar10.ckpt"):
    # instantiate model
    model = resnet18(weights=None, num_classes=NUM_CLASSES)  # Assuming CIFAR-10 dataset with 10 classes
    checkpoint = torch.load(model_ckpt, map_location="cpu")
    model.load_state_dict(checkpoint)

    # Define the target layer for Grad-CAM
    target_layers = [model.layer4[-1]]  # Last layer of the last block in ResNet18

    # TODO: Only load the subset of correctly predicted samples
    # TODO: Move this to utils.py
    # Load CIFAR10 dataset
    correct_idx = np.load(subset_idx_file)
    test_dataset = CIFAR10(root='./data', train=False, download=False, transform=data_transforms)
    data_subset = Subset(test_dataset, correct_idx)
    test_loader = torch.utils.data.DataLoader(data_subset, batch_size=4, shuffle=False)

    # Produce explanations on the test set
    all_importances = {n: None for n in XAI_METHODS.keys()}
    all_explanation_wts = {n: None for n in XAI_METHODS.keys()}
    for name, xai_method in XAI_METHODS.items():
        print(f"Computing {name} explanations...")
        if name in 'GradCAM':
            explainer = xai_method(model, target_layers=target_layers)
        else:
            explainer = xai_method(model)

        explanations = explainer.explain(test_loader)
        importance_maps, exp_flat = explainer.importance_maps(explanations)
        all_importances[name] = importance_maps
        all_explanation_wts[name] = exp_flat
        # np.save(importance_maps, f'{name}_importance.npy')
        # np.save(exp_flat, f'{name}_exp_wts.npy')

    return all_importances, all_explanation_wts

def perturbation():
    pixel_importances, exp_wts = produce_explanations()
    # for n in XAI_METHODS.keys:
    #     pass

if __name__=='__main__':
    produce_explanations()