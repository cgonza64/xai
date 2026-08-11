import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
from tqdm import tqdm
from functools import partial
import gc
import torch
from GradCamXai import GradCamXai
from IntegratedGradientsXai import IntegratedGradientsXai
from LimeXai import LimeXai
from explanations import XAI_METHODS
from utils import load_model, load_data_samples, lime_transforms, IMG_SIZE

Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def display_images(image_paths, labels=None, figsize=(12, 3), num_images=4, show=True, save_path=None):
    """
    Load 4 images and display them side-by-side with overhead labels.
    
    Args:
        image_paths (list): List of paths to the image files to be displayed.
        labels (list): List of labels for each image. If None, the image file names will be used as labels.
        figsize (tuple): Size of the figure to be displayed.
        num_images (int): Number of images to display. Must be 4.
        show (bool): Whether to display the figure.
        save_path (str): Path to save the figure.
    """
    if len(image_paths) != num_images:
        raise ValueError(f"image_paths must contain exactly {num_images} image files.")

    if labels is None:
        labels = [os.path.basename(path) for path in image_paths]

    if len(labels) != num_images:
        raise ValueError("labels must contain exactly 4 items.")

    images = [Image.open(path).convert("RGB") for path in image_paths]
    # target_size = images[0].size
    target_size = (512, 512)
    images = [img.resize(target_size, Image.Resampling.LANCZOS) for img in images]

    fig, axes = plt.subplots(1, num_images, figsize=figsize)
    for ax, img, label in zip(axes, images, labels):
        ax.imshow(img)
        ax.set_title(label)
        ax.axis("off")

    if save_path is not None:
        plt.savefig(save_path)

    if show:
        plt.tight_layout()
        plt.show()

def correct_image_samples(n_samples=5,
                             model_ckpt="ckpts/rn18_cifar10.ckpt",
                             correct_preds="ckpts/correct_preds_indices.npy",
                             outdir='./visualizations'):
    """
    Displays some the correctly predicted images from the CIFAR-10 dataset.
    These can then be used as references for generated explanation visualizations.

    Args:
        n_samples (int): Number of samples to display.
        model_ckpt (str): Path to the model checkpoint file.
        correct_preds (str): Path to the numpy file containing the indices of correctly predicted samples.
        outdir (str): Directory to save the images.
    """
    test_dataset = CIFAR10(root='./data', train=False, download=False)
    correct_indices = np.load(correct_preds)
    idx = correct_indices[:n_samples]
    save_path = f"{outdir}/correct_predictions"
    os.makedirs(save_path, exist_ok=True)
    for i in idx:
        img, tgt = test_dataset[i]
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
        img.save(f"{save_path}/img_{i}.png")

def visualize_sample_explanations(n_samples=5,
                                  methods=XAI_METHODS,
                                  model_ckpt="ckpts/rn18_cifar10.ckpt",
                                  outdir='./visualizations',
                                  show=True):
    """
    Displays explanation visualizations for each XAI method.
    
    Args:
        n_samples (int): Number of samples to generate explanations for.
        methods (list): List of XAI methods to generate explanations for.
        model_ckpt (str): Path to the model checkpoint file.
        outdir (str): Directory to save the explanations.
        show (bool): Whether to display the explanations.
    """
    print(f"\nDisplaying {n_samples} explanations for each XAI method...")
    model = load_model(model_ckpt=model_ckpt)
    explainers = {
        'GradCAM': GradCamXai(model=model, target_layers=[model.layer4[-1]]),
        'IntegratedGradients': IntegratedGradientsXai(model=model),
        'LIME': LimeXai(model=model)
    }
    test_dataset = CIFAR10(root='./data', train=False, download=False)
    class_names = test_dataset.classes
       
    # Generate visualizations for each XAI method
    for m in methods:
        print(f"\n--- {m} ---")
        save_path = f"{outdir}/{m}"
        os.makedirs(save_path, exist_ok=True)
        dloader = load_data_samples()
        data, ground_truth = next(iter(dloader))
        if m == "LIME":
            data = load_data_samples(transforms=lime_transforms, data_only=True)
        samples = data[:n_samples]
        y = ground_truth[:n_samples]

        # Force tqdm to stop rendering for every LIME explanation
        disable_progress_bar = True if m == 'LIME' else False
        tqdm.__init__ = partial(tqdm.__init__, disable=disable_progress_bar)

        print(f"Generating {n_samples} explanations...")
        exp_method = explainers[m]
        explanations = exp_method.explain(samples, labels=y)
        if m == "LIME":
            _, visual_info = explanations  # Only the visual info is needed for LIME
            explanations = list(range(n_samples))

        for i, e in enumerate(explanations):
            print(f"Showing explanation {i+1}")
            if m == 'LIME':
                img, mask, target_label = [visual_info[k][i] for k in visual_info.keys()]
                exp_method.visualize(img,
                                     mask,
                                     class_names[target_label],
                                     show=show,
                                     save_path=f"{save_path}/{m}_exp_{i}.png")
            else:
                target_label = y[i].item()
                exp_method.visualize(samples[i],
                                     e,
                                     class_names[target_label],
                                     show=show,
                                     save_path=f"{save_path}/{m}_exp_{i}.png")

        # Free up RAM/VRAM for next method
        del explanations, exp_method
        gc.collect()
        torch.cuda.empty_cache()

def print_model_confidence(n_samples=5,
                           model_ckpt="ckpts/rn18_cifar10.ckpt",
                           correct_preds="ckpts/correct_preds_indices.npy"):
    """
    Helper function to print a model's prediction confidence for a number of correctly predicted 
    CIFAR-10 test samples.

    Args:
        n_samples (int): Number of samples to evaluate.
        model_ckpt (str): Path to the model checkpoint file.
        correct_preds (str): Path to the numpy file containing the indices of correctly predicted samples.
    """
    tqdm.__init__ = partial(tqdm.__init__, disable=False)  # re-enable TQDM output
    print(f"\n--- Extracting Model Confidence for {n_samples} samples ---")
    dloader = load_data_samples()
    model = load_model(model_ckpt=model_ckpt)
    model.to(Device)
    test_dataset = CIFAR10(root='./data', train=False, download=False)
    class_names = test_dataset.classes
    correct_idx = np.load(correct_preds)
    targets = []
    confidences = []
    with torch.no_grad():
        for imgs, labels in tqdm(dloader, desc="Extracting Prediction Confidences"):
            imgs = imgs.to(Device)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=-1)
            preds = probs.max(dim=-1).values
            confidences.append(preds.detach().cpu().numpy())
            targets.extend(labels.detach().cpu().numpy().tolist())

    confidences = np.concatenate(confidences)
    for i in range(n_samples):
        idx = correct_idx[i]
        pred = targets[i]
        print(f"idx={idx}, class={class_names[pred]}, confidence={confidences[i]:.4f}")

def visual_comparison(idx=[2, 3, 4, 6, 8, 9, 10, 11, 16, 17, 22, 25], img_dir="./sample_explanations"):
    """
    Helper function to save a side-by-side comparison of original input images and their corresponding explanation visualizations
    generated by GradCAM, Integrated Gradients, and LIME.The images are saved in a structured directory format for easy comparison.
    
    Args:
        idx (list): List of indices of the samples to be visualized.
        img_dir (str): Directory to save the comparison images.
    """
    test_dataset = CIFAR10(root='./data', train=False, download=False)
    class_labels = test_dataset.classes
    for i in idx:
        save_path = f"{img_dir}/img_{i}"
        paths = [
            f"{save_path}/img_{i}.png",
            f"{save_path}/GradCAM_exp_{i}.png",
            f"{save_path}/IntegratedGradients_exp_{i}.png",
            f"{save_path}/LIME_exp_{i}.png",
        ]

        _, target = test_dataset[i]
        classname = class_labels[target]
        labels = [f"Original - '{classname}'", 'GradCAM', 'Integrated Gradients', 'LIME',]
        display_images(image_paths=paths,
                       labels=labels,
                       figsize=(11, 5),
                       num_images=len(paths),
                       show=False,
                       save_path=f"{save_path}/img_{i}_comparison.png")

if __name__=='__main__':
    number_of_samples = 5

    # Display some of the correct predictions from the CIFAR-10 test dataset
    correct_image_samples(n_samples=number_of_samples)

    # Generate samples explanations for visual comparison
    visualize_sample_explanations(n_samples=number_of_samples,
                                  model_ckpt="ckpts/rn18_cifar10.ckpt",
                                  show=True)

    # Model Confidences
    print_model_confidence(n_samples=number_of_samples)

    # Plot/Save explanation visuals side-by-side for comparison
    # visual_comparison()
