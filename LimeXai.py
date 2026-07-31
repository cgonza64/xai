from XaiMethodBase import XaiMethodBase
import matplotlib.pyplot as plt
from PIL import Image
from utils import lime_transforms, load_model
from utils import CIFAR10_MEAN, CIFAR10_STD, BATCH_SIZE
from tqdm import tqdm
import numpy as np
import time
import sys

# Torch
import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torch.utils.data import Subset
import cv2

# LIME

from lime import lime_image
from skimage.segmentation import mark_boundaries
from functools import partial

class LimeXai(XaiMethodBase):
    """
    LIME implementation for generating visual explanations.
    """

    def __init__(self, model, batch_size=4):
        super().__init__(model)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size

    def batch_predict_cifar10(self, images):
        """ Called by the LIME explainer to processes image batches."""
        self.model.eval()
        batch = torch.tensor(images, dtype=torch.float32, device=self.device)
        batch = batch.permute(0, 3, 1, 2) / 255.0
        
        # Vectorized normalization on the GPU
        cifar_mean = torch.tensor(CIFAR10_MEAN, device=self.device).view(1, 3, 1, 1)
        cifar_std = torch.tensor(CIFAR10_STD, device=self.device).view(1, 3, 1, 1)
        batch = (batch - cifar_mean) / cifar_std

        with torch.no_grad():
            logits = self.model(batch)
            probs = F.softmax(logits, dim=1)
            
        return probs.cpu().numpy()

    def explain(self, images, labels=None, info=False):
        """
        Generate LIME explanations for the given input data.

        Args:
            images: The input images for which explanations are to be generated.

        Returns:
            A dictionary of LIME superpixel weights and their corresponding segment maps.
        """
        explanations = {'segments': [], 'weights': []}
        visuals = {'images': [], 'masks': [], 'top_labels': []}
        self.model.to(self.device)
        self.model.eval()
        explainer = lime_image.LimeImageExplainer()

        # Generate LIME explanations
        N = len(images)
        start = time.time()  # Measure total duration
        for idx, img in enumerate(images):
            explanation = explainer.explain_instance(
                img, 
                self.batch_predict_cifar10,
                top_labels=1,
                num_samples=500, # Keeps execution fast
                batch_size=self.batch_size,
            )

            # save explanation weights and superpixel segments
            explanations['segments'].append(explanation.segments)
            top_label = explanation.top_labels[0]
            explanations['weights'].append(explanation.local_exp[top_label])

            # save feature importance mask and corresponding label for visualization purposes
            top_label = explanation.top_labels[0]
            temp, mask = explanation.get_image_and_mask(
                top_label,
                positive_only=False,
                num_features=5,
                hide_rest=False
            )
            visuals['images'].append(temp)
            visuals['masks'].append(mask)
            visuals['top_labels'].append(top_label)

            sys.stderr.write(f"\rExplanations: {idx+1}/{N}")
            sys.stderr.flush()

        time_elapsed = time.time() - start
        print()
        if info:
            print(f"\nGenerated {N} explanations in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")

        return explanations, visuals

    def visualize(self, img, mask, target_class, show=True, save_path=None):
        """
        Visualize the LIME explanation overlayed onto the input image.
        """
        fig, ax = plt.subplots(1, 1)
        visualization = mark_boundaries(img / 255.0, mask)
        ax.imshow(visualization)
        ax.set_title(f"LIME - Target Class: {target_class}")
        ax.axis('off')

        # Save image to file
        if save_path is not None:
            image = (visualization * 255).astype(np.uint8)
            cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        if show:
            plt.tight_layout()
            plt.show()

    def importance_maps(self, explanations):
        """
        Convert LIME explanations to pixel importance maps.
        """
        segments = explanations['segments']
        weights = explanations['weights']

        # Convert per-pixel attributions to pixel importance mapping
        all_attributions = []
        for seg, wts in zip(segments, weights):
            attributions = np.zeros_like(seg, dtype=np.float32)

            # Assign each pixel the same weight as its superpixel
            for superpixel_id, weight in wts:
                attributions[seg == superpixel_id] = weight

            # Normalize
            attributions = (attributions - attributions.min()) / (attributions.max() - attributions.min() + 1e-8)
            all_attributions.append(attributions)

        return np.array(all_attributions)

if __name__ == "__main__":
    # Force tqdm to stop rendering for every LIME explanation
    tqdm.__init__ = partial(tqdm.__init__, disable=True)

    # Load a pre-trained model (ResNet18)
    model = load_model(model_ckpt="ckpts/rn18_cifar10.ckpt")

    # Create an instance of LIME
    lime = LimeXai(model)

    # Load CIFAR10 dataset
    test_dataset = CIFAR10(root='./data', train=False, download=False, transform=lime_transforms)
    subset = Subset(test_dataset, list(range(10)))
    images = [subset[_][0] for _ in range(len(subset))]
    images = np.array(images)

    # Generate explanations for the test dataset
    explanations, visual_info = lime.explain(images, info=True)

    # Visualize the first explanation
    tgt_classes = test_dataset.classes
    img, mask, target_label = [visual_info[k][0] for k in visual_info.keys()]
    lime.visualize(img, mask, tgt_classes[target_label], save_path='lime_output.png')

    # test image
    test_image = "./sample_explanations/cat_0.jpg"
    img = Image.open(test_image)
    img = lime_transforms(img)
    explanation, visual_info = lime.explain(np.array([img]))  # Assuming label 3 for cat
    img, mask, target_label = [visual_info[k][0] for k in visual_info.keys()]
    lime.visualize(img, mask, tgt_classes[target_label], save_path='lime_cat.png')

    # importance maps and feature rankings
    print()
    importance_maps = lime.importance_maps(explanations)
    bs = len(explanations['weights'])
    im_flat = importance_maps.reshape(bs, -1)
    feature_rankings = np.argsort(im_flat)[:, ::-1]
    print(type(importance_maps), type(feature_rankings))
    print(importance_maps.shape, feature_rankings.shape)
    print(im_flat[0][feature_rankings[0][:5]])  # Should be in descending order
