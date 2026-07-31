from GradCamXai import GradCamXai
from IntegratedGradientsXai import IntegratedGradientsXai
from LimeXai import LimeXai
from utils import load_data_samples, load_model, data_transforms, lime_transforms
from tqdm import tqdm
from functools import partial
import numpy as np
import os
import time
import gc
import torch

# Number of explanation collection rounds per XAI method
# At least 5 for statistical signifance purposes
NUMBER_OF_ITERATIONS = 5

# Explanation Collection Configurations for each XAI method
XAI_METHODS = {
    'GradCAM': {
        'ctor': GradCamXai,
        'batch_size': 128,
        'data_only': False,
        'data_transforms': data_transforms
    },
    'IntegratedGradients': {
        'ctor': IntegratedGradientsXai,
        'batch_size': 2,
        'data_only': False,
        'data_transforms': data_transforms
    },
    'LIME': {
        'ctor': LimeXai,
        'batch_size': 128,
        'data_only': True,
        'data_transforms': lime_transforms
    }
}

def produce_explanations(model_ckpt="ckpts/rn18_cifar10.ckpt", save_path='explanations', n_iter=NUMBER_OF_ITERATIONS):
    """ TODO: add description. """
    # instantiate model
    model = load_model(model_ckpt=model_ckpt)

    # Define the target layer for Grad-CAM
    target_layers = [model.layer4[-1]]  # Last layer of the last block in ResNet18

    # Produce explanations for each XAI method
    for name, config in XAI_METHODS.items():
        print(f"\n--- {name} ---")
        stats_path = f"{save_path}/{name}"
        os.makedirs(stats_path, exist_ok=True)

        # Load the correctly predicted data samples from the CIFAR10 testset
        bs, data_only, transforms = config['batch_size'], config['data_only'], config['data_transforms']
        dloader = load_data_samples(batch_size=bs, data_only=data_only, transforms=transforms)

        N = len(dloader) if name == "LIME" else len(dloader.dataset)

        # Force tqdm to stop rendering for every LIME explanation
        disable_progress_bar = True if name == 'LIME' else False
        tqdm.__init__ = partial(tqdm.__init__, disable=disable_progress_bar)

        if name == 'GradCAM':
            explainer = config['ctor'](model, target_layers=target_layers)
        else:
            explainer = config['ctor'](model)

        # Perform multiple executions to determine statistical siginificance
        for n in range(n_iter):
            print(f"iteration {n+1}:")

            # Explanation generation duration
            start = time.time()  

            # Generate explanations and extract feature importance mappings
            explanations = explainer.explain(dloader)
            time_elapsed = time.time() - start
            print(f"Generated {N} explanations in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")

            print("Extracting feature importance mappings...")
            explanations = explanations[0] if name == "LIME" else explanations
            importance_mappings = explainer.importance_maps(explanations)
            
            # Save for later processing
            print("Saving to disk...")
            np.save(f"{stats_path}/importance_maps_{n+1}.npy", importance_mappings)
            np.save(f"{stats_path}/duration_{n+1}.npy", time_elapsed)

            # Free up RAM/VRAM for next iteration
            del explanations, importance_mappings
            gc.collect()
            torch.cuda.empty_cache()

        # Free up RAM/VRAM for next method
        del explainer
        gc.collect()
        torch.cuda.empty_cache()

if __name__=='__main__':
    produce_explanations()