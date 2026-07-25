from GradCamXai import GradCamXai
from utils import load_data_samples, load_model, NUM_CLASSES
import numpy as np
import os
import time
import torch
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
from torch.utils.data import Subset

XAI_METHODS = {
    'GradCAM': GradCamXai,
    # 'IntegratedGradients': IGXai,
    # 'LIME': LimeXai
}

def produce_explanations(model_ckpt="ckpts/rn18_cifar10.ckpt", explanations_path='explanations'):
    # instantiate model
    model = load_model(model_ckpt=model_ckpt)

    # Define the target layer for Grad-CAM
    target_layers = [model.layer4[-1]]  # Last layer of the last block in ResNet18

    # Load the correctly predicted data samples from the CIFAR10 testset
    dloader = load_data_samples()
    N = len(dloader.dataset)

    # Produce explanations on the test set
    all_importance_maps = {n: None for n in XAI_METHODS.keys()}
    all_explanation_wts = {n: None for n in XAI_METHODS.keys()}
    all_durations = {n: None for n in XAI_METHODS.keys()}
    for name, xai_method in XAI_METHODS.items():
        print(f"\n--- {name} ---")
        stats_path = f"{explanations_path}/{name}"
        os.makedirs(stats_path, exist_ok=True)

        # Explanation generation duration
        start = time.time()  

        if name in 'GradCAM':
            explainer = xai_method(model, target_layers=target_layers)
        else:
            explainer = xai_method(model)

        # Generate explanations and extract feature importance mappings
        explanations = explainer.explain(dloader)
        time_elapsed = time.time() - start
        all_durations[name] = time_elapsed
        print(f"Generated {N} explanations in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")

        print("Extracting feature importance mappings...")
        importance_mappings = explainer.importance_maps(explanations)
        all_importance_maps[name] = importance_mappings
        
        # Save for later processing
        print("Saving to disk...")
        np.save(f"{stats_path}/importance_maps.npy", importance_mappings)
        np.save(f"{stats_path}/duration.npy", time_elapsed)

    return all_importance_maps, all_explanation_wts

def perturbation(model_ckpt="ckpts/rn18_cifar10.ckpt", explanations_path='explanations'):
    # instantiate model
    model = load_model(model_ckpt=model_ckpt)

    # Load the correctly predicted data samples from the CIFAR10 testset
    dloader = load_data_samples()
    N = len(dloader.dataset)

    # Produce explanations on the test set
    all_importance_maps = {n: None for n in XAI_METHODS.keys()}
    all_explanation_wts = {n: None for n in XAI_METHODS.keys()}
    all_durations = {n: None for n in XAI_METHODS.keys()}
    for name, xai_method in XAI_METHODS.items():
        print(f"\n--- {name} ---")
        stats_path = f"{explanations_path}/{name}"
        importance_mappings = np.load(f"{stats_path}/importance_mappings.npy")
        explanation_wts = np.load(f"{stats_path}/explanation_wts.npy")

if __name__=='__main__':
    produce_explanations()