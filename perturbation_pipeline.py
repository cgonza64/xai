
import torch
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms.functional as TF
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import os
from utils import IMG_SIZE, CIFAR10_MEAN
from utils import load_data_samples, load_model, tensor2image
from explanations import XAI_METHODS, NUMBER_OF_ITERATIONS

PERTURBATION_RATIOS = torch.linspace(0,1,11)
Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_masks(indices, fraction, H=IMG_SIZE, W=IMG_SIZE):
    """
    Helper function to mask a fraction of the important input features 
    as determined by the given explanation feature rankings.

    Args:
        indices (torch.Tensor): Tensor of shape (batch_size, num_features) containing the feature rankings for each sample in the batch.
        fraction (float): Fraction of the important features to mask.
        H (int): Height of the input image.
        W (int): Width of the input image.
    Returns:
        torch.Tensor: Tensor of shape (batch_size, H, W) containing the binary masks.
    """
    bs = indices.shape[0]
    total = indices.shape[1]
    k = int(total*fraction)
    masks = torch.ones(bs, total, device=Device)
    if k != 0:
        masks.scatter_(dim=1, index=indices[:, :k], value=0)
    return masks.reshape(bs, H, W)

def batched_AUC(fractions, batched_scores):
    """
    Helper function to compute feature deletion/insertion AUC for a batch of model confidence scores.

    Args:
        fractions (torch.Tensor): Tensor of shape (num_ratios,) containing the perturbation ratios.
        batched_scores (np.ndarray): Array of shape (num_ratios, batch_size) containing the model confidence scores for each sample in the batch.
    Returns:
        np.ndarray: Array of shape (batch_size,) containing the AUC for each sample in the batch.
    """
    all_aucs = []
    n_samples = batched_scores.shape[1]
    for i in range(n_samples):
        all_aucs.append(auc(fractions, batched_scores[:, i]))
    return np.array(all_aucs)

def perturbation_curve(model, images, targets, feature_importance, mode='deletion', fractions=torch.linspace(0,1,6)):
    """
    Computes model confidence scores while varying the number of important pixels for a batch of input images.
    The important pixels are based on the results of a model explainer method (e.g., GradCAM, Integrated Gradients, LIME, etc).

    Args:
        model (torch.nn.Module): Pre-trained model to evaluate.
        images (torch.Tensor): Tensor of shape (batch_size, channels, height, width) containing the input images.
        targets (torch.Tensor): Tensor of shape (batch_size,) containing the true labels for each image.
        feature_importance (torch.Tensor): Tensor of shape (batch_size, height, width) containing the feature importance scores for each image.
        mode (str): Either 'deletion' or 'insertion' to specify the type of perturbation.
        fractions (torch.Tensor): Tensor of shape (num_ratios,) containing the perturbation ratios.
    """
    H, W = feature_importance.shape[-2:]
    
    # Process feature importance mappings
    importance_flat = feature_importance.flatten(start_dim=1)
    feature_rankings = torch.argsort(importance_flat, descending=True)

    # # Alternate: Baselines using transformed versions of each image in the batch
    # baselines = TF.gaussian_blur(img=images, kernel_size=21, sigma=0.5)
    # baselines = TF.adjust_brightness(img=ins_baselines, brightness_factor=0.3)

    # Mean baseline
    baselines = torch.tensor(CIFAR10_MEAN, device=Device).view(3, 1, 1).expand_as(images)

    # Compute model confidence scores while varying the number of important input features
    scores=[]
    for f in fractions:
        masks = create_masks(feature_rankings, f, H, W).unsqueeze(1)
        if mode == 'deletion':
            perturbed = baselines * (1 - masks) + images * masks
        elif mode == 'insertion':
            perturbed = baselines * masks + images * (1 - masks)
        else:
            raise ValueError(f"mode must be either 'deletion' or 'insertion', but '{mode}' was received instead.")

        # # Alternate: Use zero-intensity for masked pixels
        # perturbed=images.clone()
        # perturbed *= masks.unsqueeze(1)

        perturbed = perturbed.to(Device)
        with torch.no_grad():
            preds=model(perturbed)
        prob=torch.softmax(preds,1)
        confidences = torch.gather(prob, dim=1, index=targets.unsqueeze(1)).squeeze(1)
        scores.append(confidences.detach().cpu().numpy())

    return fractions, np.vstack(scores)

def pipeline(model_ckpt,
             data_idx,
             ratios=torch.linspace(0,1,6),
             explanations_pth="./explanations",
             perturbation_scores_pth="./perturbation_scores",
             methods=list(XAI_METHODS.keys())):
    """
    Runs the perturbation pipeline for each XAI method and saves the results to disk.

    Args:
        model_ckpt (str): Path to the pre-trained model checkpoint.
        data_idx (str): Path to the file containing the indices of correctly predicted samples.
        ratios (torch.Tensor): Tensor of shape (num_ratios,) containing the perturbation ratios.
        explanations_pth (str): Path to the directory containing the explanation maps.
        perturbation_scores_pth (str): Path to the directory where perturbation scores will be saved.
        methods (list): List of XAI methods for which to run the perturbation pipeline.
    """
    print("Running Perturbation Pipeline for each XAI method...")
    # Instantiate model
    model = load_model(model_ckpt=model_ckpt)

    # Load the correctly predicted data samples from the CIFAR10 testset
    dloader = load_data_samples(correct_indices_pth=data_idx)

    # Compute feature perturbation scores for each XAI method
    for m in methods:
        print(f"\n--- {m} ---")
        save_path = f"{perturbation_scores_pth}/{m}"
        os.makedirs(save_path, exist_ok=True)
        stats_path = f"{explanations_pth}/{m}"

        # Compute scores for each explanation collection iteration
        method_deletion_auc, method_insertion_auc = [], []
        avg_del_confidence, avg_ins_confidence = [], []
        for i in range(1, NUMBER_OF_ITERATIONS+1):
            importance_maps = np.load(f"{stats_path}/importance_maps_{i}.npy")

            bs = importance_maps.shape[0]
            exp_dataset = TensorDataset(torch.tensor(importance_maps), torch.zeros(bs))  # the targets don't matter here
            exp_dataloader = DataLoader(exp_dataset, batch_size=dloader.batch_size)

            # Prepare model for inferencing
            model.eval()
            model.to(Device)

            feat_del_scores, feat_ins_scores = [], []
            for input_data, explanations in tqdm(zip(dloader, exp_dataloader), desc=f"Confidence Scores {i}"):
                images, targets = input_data
                importance_mappings, _ = explanations
                images = images.to(Device)
                targets = targets.to(Device)
                importance_mappings = importance_mappings.to(Device)

                # feature deletion            
                _, scores = perturbation_curve(model=model,
                                            images=images,
                                            targets=targets,
                                            feature_importance=importance_mappings,
                                            fractions=ratios)
                feat_del_scores.append(scores)

                # feature insertion
                _, scores = perturbation_curve(model=model,
                                            images=images,
                                            targets=targets,
                                            feature_importance=importance_mappings,
                                            mode='insertion',
                                            fractions=ratios)
                feat_ins_scores.append(scores)

            # Compute per-instance AUC for this iteration
            feat_del_scores = np.hstack(feat_del_scores)
            feat_ins_scores = np.hstack(feat_ins_scores)
            feat_deletion_auc = batched_AUC(fractions=ratios, batched_scores=feat_del_scores)
            feat_insertion_auc = batched_AUC(fractions=ratios, batched_scores=feat_ins_scores)
            method_deletion_auc.append(feat_deletion_auc)
            method_insertion_auc.append(feat_insertion_auc)

            # save average model confidence scores per ratio
            avg_del_confidence.append(np.mean(feat_del_scores, axis=1))
            avg_ins_confidence.append(np.mean(feat_ins_scores, axis=1))

        # Save all AUC scores for this XAI method
        method_deletion_auc = np.vstack(method_deletion_auc)
        method_insertion_auc = np.vstack(method_insertion_auc)
        np.save(f"{save_path}/feature_deletion_auc.npy", method_deletion_auc)
        np.save(f"{save_path}/feature_insertion_auc.npy", method_insertion_auc)

        # Average per-instance AUC over each iteration before computing total average AUC
        avg_del_scores = np.mean(method_deletion_auc, axis=0)
        print(f"Avg Feature Deletion AUC={np.mean(avg_del_scores):.3f} {chr(177)}{np.std(avg_del_scores):.4f}")
        avg_ins_scores = np.mean(method_insertion_auc, axis=0)
        print(f"Avg Feature Insertion AUC={np.mean(avg_ins_scores):.3f} {chr(177)}{np.std(avg_ins_scores):.4f}")

        # Save average model confidence vs perturbation ratios
        avg_del_confidence = np.vstack(avg_del_confidence)
        avg_ins_confidence = np.vstack(avg_ins_confidence)
        np.save(f"{save_path}/deletion_confidence.npy", avg_del_confidence)
        np.save(f"{save_path}/insertion_confidence.npy", avg_ins_confidence)

def visualize_perturbations(img, mask, ratio=0.3, mode='deletion'):
    """
    Helper function to visualize example images side-by-side with their perturbed versions.

    Args:
        img (torch.Tensor): Tensor of shape (1, channels, height, width) containing the input image.
        mask (torch.Tensor): Tensor of shape (1, 1, height, width) containing the binary mask for the important features.
        ratio (float): Fraction of the important features that were perturbed.
        mode (str): Either 'deletion' or 'insertion' to specify the type of perturbation.
    """
    baseline = TF.gaussian_blur(img=img, kernel_size=21, sigma=2.5)
    baseline = TF.adjust_brightness(img=baseline, brightness_factor=0.3)

    # mean baseline
    baseline = torch.tensor(CIFAR10_MEAN, device=Device).view(3, 1, 1).expand_as(img)

    if mode in 'deletion':
        perturbed = baseline * (1 - mask) + img * mask
    elif mode == 'insertion':
        perturbed = baseline * mask + img * (1 - mask)
    else:
        raise ValueError(f"mode must be either 'deletion' or 'insertion', but '{mode}' was received instead.")

    og_and_blurred = torch.cat((img, perturbed), dim=0)
    images = [tensor2image(inp.cpu()) for inp in og_and_blurred]

    num_images = len(images)
    fig, axes = plt.subplots(1, num_images)
    for ax, img, label in zip(axes, images, ['Original', 'Perturbed']):
        ax.imshow(img)
        ax.set_title(f"{label}")
        ax.axis("off")

    plt.suptitle(f"Feature {mode}: {100*ratio} %")
    plt.tight_layout()
    plt.show()


if __name__=='__main__':
    print(torch.cuda.get_device_name(0))

    # Compute and save the pertubation scores for each XAI method
    pipeline(model_ckpt="ckpts/rn18_cifar10.ckpt",
             ratios=PERTURBATION_RATIOS,
             data_idx="./ckpts/correct_preds_indices.npy")

    # # Visualize a sample image along with its perturbed version
    # dloader = load_data_samples(correct_indices_pth="./ckpts/correct_preds_indices.npy")
    # first_batch, _ = next(iter(dloader))
    # test_image = first_batch[0].unsqueeze(0).to(Device)
    # importance_maps = np.load("explanations/LIME/importance_maps.npy")
    # importance_maps = torch.tensor(importance_maps)
    # importance_flat = importance_maps.flatten(start_dim=1)
    # feature_rankings = torch.argsort(importance_flat, descending=True)
    # first_sample = feature_rankings[0].unsqueeze(0).to(Device)
    # H, W = importance_maps.shape[-2:]
    # r = 0.4  # perturbation ratio
    # masks = create_masks(first_sample, r, H, W).unsqueeze(1)
    # visualize_perturbations(test_image, masks, ratio=r, mode='deletion')
    # visualize_perturbations(test_image, masks, ratio=r, mode='insertion')
