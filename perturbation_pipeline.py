
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
from explanations import XAI_METHODS

Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_masks(indices, fraction, H=IMG_SIZE, W=IMG_SIZE):
    """
    Helper function to mask a fraction of the important input features 
    as determined by the given explanation feature rankings.
    """
    bs = indices.shape[0]
    total = indices.shape[1]
    k = int(total*fraction)
    masks = torch.ones(bs, total, device=Device)
    if k != 0:
        masks.scatter_(dim=1, index=indices[:, :k], value=0)
    return masks.reshape(bs, H, W)

def batched_AUC(fractions, batched_scores):
    """ Helper function to compute feature deletion/insertion AUC for a batch of model confidence scores. """
    all_aucs = []
    n_samples = batched_scores.shape[1]
    for i in range(n_samples):
        all_aucs.append(auc(fractions, batched_scores[:, i]))
    return all_aucs

def perturbation_curve(model, images, targets, feature_importance, mode='deletion', fractions=torch.linspace(0,1,6)):
    """
    Computes model confidence scores while varying the number of important pixels for a batch of input images.
    The important pixels are based on the results of a model explainer method (e.g., GradCAM, Integrated Gradients, LIME, etc).
    """
    H, W = feature_importance.shape[-2:]    
    
    # Process feature importance mappings
    importance_flat = feature_importance.flatten(start_dim=1)
    feature_rankings = torch.argsort(importance_flat, descending=True)

    # # Baselines using transformed versions of each image in the batch
    # baselines = TF.gaussian_blur(img=images, kernel_size=21, sigma=0.5)
    # baselines = TF.adjust_brightness(img=baselines, brightness_factor=0.3)

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
    """ TODO: add description. """
    print("Running Perturbation Pipeline for each XAI methods...")
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
        importance_maps = np.load(f"{stats_path}/importance_maps.npy")
        print(importance_maps.shape)

        bs = importance_maps.shape[0]
        exp_dataset = TensorDataset(torch.tensor(importance_maps), torch.zeros(bs))  # the targets don't matter here
        exp_dataloader = DataLoader(exp_dataset, batch_size=dloader.batch_size)

        # Prepare model for inferencing
        model.eval()
        model.to(Device)

        feat_del_scores, feat_ins_scores = [], []
        for input_data, explanations in tqdm(zip(dloader, exp_dataloader), desc="Computing Scores"):
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

        feat_del_scores = np.hstack(feat_del_scores)
        feat_ins_scores = np.hstack(feat_ins_scores)
        np.save(f"{save_path}/feature_deletion_scores.npy", feat_del_scores)
        np.save(f"{save_path}/feature_insertion_scores.npy", feat_ins_scores)

        feat_deletion_auc = batched_AUC(fractions=ratios, batched_scores=feat_del_scores)
        print(f"Avg Feature Deletion AUC={np.mean(feat_deletion_auc):.3f} {chr(177)}{np.std(feat_deletion_auc):.4f}")
        feat_insertion_auc = batched_AUC(fractions=ratios, batched_scores=feat_ins_scores)
        print(f"Avg Feature Insertion AUC={np.mean(feat_insertion_auc):.3f} {chr(177)}{np.std(feat_insertion_auc):.4f}")

def visualize_perturbations(input, masks):
    """ Function to visualize example pertubations side-by-side with their unperturbed versions. """
    baselines = TF.gaussian_blur(img=input, kernel_size=21, sigma=2.5)
    baselines = TF.adjust_brightness(img=baselines, brightness_factor=0.3)

    # mean baseline
    baselines = torch.tensor(CIFAR10_MEAN).view(3, 1, 1).expand_as(input)

    og_img = input[0].unsqueeze(0)
    perturbed = baselines * (1 - masks.unsqueeze(1)) + input * masks.unsqueeze(1)
    og_and_blurred = torch.cat((og_img, perturbed), dim=0)
    images = [tensor2image(inp) for inp in og_and_blurred]

    num_images = len(images)
    fig, axes = plt.subplots(1, num_images, figsize=(12, 3))
    for ax, img, label in zip(axes, images, np.arange(1, num_images+1)):
        ax.imshow(img)
        ax.set_title(f"image {label}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__=='__main__':
    print(torch.cuda.get_device_name(0))

    pipeline(model_ckpt="ckpts/rn18_cifar10.ckpt", data_idx="./ckpts/correct_preds_indices.npy")

    # # instantiate model
    # model = load_model(model_ckpt="ckpts/rn18_cifar10.ckpt")

    # # Load the correctly predicted data samples from the CIFAR10 testset
    # dloader = load_data_samples()
    # test_data, targets = next(iter(dloader))
    # N = len(dloader.dataset)

    # stats_path = f"./explanations/GradCAM"
    # # importance_rankings = np.load(f"{stats_path}/importance_mapping.npy")
    # # importance_rankings = torch.tensor(importance_rankings)
    # explanation_wts = np.load(f"{stats_path}/explanation_wts.npy")
    # print(f"explanations={explanation_wts.shape}")
    # explanation_wts = explanation_wts.reshape(-1, IMG_SIZE, IMG_SIZE)
    # # print('creating masks')
    # # masks = create_masks(importance_rankings[:100], fraction=0.5)
    # # print(masks[0].sum())
    # # for i in tqdm(np.arange(0, len(importance_rankings), 1000)):
    # #     idx = min(i, len(importance_rankings))
    # #     masks = create_masks(importance_rankings[idx:idx+1000], fraction=0.5)
    # #     print(masks.shape)

    # # TODO: Try computing these metrics in batches for all correctly predicted samples
    # # feature deletion
    # print('--- Feature Deletion ---')
    # fractions, scores = perturbation_curve(model, test_data, targets, explanation_wts[:BATCH_SIZE])
    # print(fractions.shape, scores.shape)
    # print(f"perturbation fractions={fractions}")
    # # feat_deletion_auc = batched_AUC(fractions=fractions, batched_scores=scores)
    # # print(f"Average Feature Deletion AUC={np.mean(feat_deletion_auc):.3f} {chr(177)}{np.std(feat_deletion_auc):.4f}")
    

    # # feature insertion
    # print('--- Feature Insertion ---')
    # fractions, scores = perturbation_curve(model, test_data, targets, explanation_wts[:BATCH_SIZE], mode='insertion')
    # # feat_insertion_auc = batched_AUC(fractions=fractions, batched_scores=scores)
    # # print(f"Average Feature Insertion AUC={np.mean(feat_insertion_auc):.3f} {chr(177)}{np.std(feat_insertion_auc):.4f}")

    # using a baseline instead of zero-intensity
    # test_image = Image.open('./sample_explanations/cat_0.jpg').convert("RGB")
    # # feature_rankings = np.expand_dims(np.arange(1, IMG_SIZE*IMG_SIZE), 0)
    # feature_rankings = torch.arange(1, IMG_SIZE*IMG_SIZE + 1).unsqueeze(0)
    # print(feature_rankings.shape)
    # dummy_mask = create_masks(feature_rankings, 0.5)
    # masks = torch.cat((dummy_mask, 1 - dummy_mask), 0)
    # input = data_transforms(test_image).unsqueeze(0) #.repeat(2, 1, 1, 1)
    # input = torch.cat((input, input), 0)
    # input = test_data[:2]
    # print(input.shape)
    # # dummy_masks = torch.ones_like(input)
    # visualize_perturbations(input, masks)