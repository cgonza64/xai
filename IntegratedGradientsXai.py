from XaiMethodBase import XaiMethodBase
import matplotlib.pyplot as plt
from PIL import Image
from utils import data_transforms, load_model, CIFAR10_MEAN, CIFAR10_STD
from tqdm import tqdm
import numpy as np

# Torch
import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import Subset

# Integrated Gradients
from captum.attr import IntegratedGradients

class IntegratedGradientsXai(XaiMethodBase):
    """
    Integrated Gradients implementation for generating visual explanations.
    """

    def __init__(self, model):
        super().__init__(model)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def explain(self, input_data, labels=None):
        """
        Generate Integrated Gradients explanations for the given input data.

        Args:
            input_data: The input data for which explanations are to be generated.

        Returns:
            Integrated Gradients saliency maps for the input data.
        """
        explanations, deltas = [], []
        self.model.to(self.device)
        self.model.eval()
        ig = IntegratedGradients(self.model)

        # Compute attributions for each input sample
        if isinstance(input_data, torch.utils.data.DataLoader):
            for imgs, _ in tqdm(input_data, desc="Generating IG explanations"):
                imgs = imgs.to(self.device)
                with torch.no_grad():
                    logits = self.model(imgs)
                    pred_classes = logits.argmax(dim=1)   # shape [B]

                # Baseline input: zero-intensity images
                baseline = torch.zeros_like(imgs).to(self.device)

                attributions, delta = ig.attribute(
                    inputs=imgs,
                    baselines=baseline,
                    target=pred_classes,
                    n_steps=50,
                    return_convergence_delta=True
                )
                explanations.append(attributions)
                deltas.append(delta)
        # Single input explanation
        elif isinstance(input_data, torch.Tensor):
            assert labels is not None, "Labels must be provided when input_data is a Tensor."
            baseline = torch.zeros_like(input_data).to(self.device)
            img = input_data.to(self.device)
            with torch.no_grad():
                logits = self.model(img)
                pred_class = logits.argmax(dim=1)   # shape [B]

            attributions, delta = ig.attribute(
                inputs=img,
                baselines=baseline,
                target=pred_class,
                n_steps=50,
                return_convergence_delta=True
            )
            explanations.append(attributions)
            deltas.append(delta)
        else:
            raise ValueError("Input data must be a DataLoader or a Tensor.")

        explanations = torch.cat(explanations, dim=0)
        deltas = torch.cat(deltas, dim=0)
        print("Mean convergence delta:", deltas.abs().mean().item())
        return explanations.detach().cpu().numpy()

    # Undo normalization for display
    @staticmethod
    def unnormalize(img_tensor):
        """ Helper function to undo normalization for display. """
        img = img_tensor.squeeze(0).cpu().clone()
        for c in range(3):
            img[c] = img[c] * CIFAR10_STD[c] + CIFAR10_MEAN[c]
        img = torch.clamp(img, 0, 1)
        return img.permute(1, 2, 0).numpy()
    
    def visualize(self, input_tensor, attributions, target_class, show=True, save_path=None):
        """
        Visualize the Integrated Gradients heatmap on the input image.
        """
        # Convert attribution to grayscale heatmap
        # attr = attributions.detach().cpu().numpy()  # shape=[3, 224, 224]
        attr = np.transpose(attributions, (1, 2, 0))  # shape=[224, 224, 3]
        attr_sum = np.sum(np.abs(attr), axis=2)  # shape=[224, 224, 1]

        orig_img = self.unnormalize(input_tensor)

        # Original image
        fig, ax = plt.subplots(1, 1)
        ax.imshow(orig_img)
        ax.imshow(attr_sum, cmap='hot', alpha=0.5)
        ax.set_title(f"Integrated Gradients - Target Class: {target_class}")
        ax.axis("off")

        if save_path is not None:
            fig.savefig(save_path)

        if show:
            plt.tight_layout()
            plt.show()

    def importance_maps(self, explanations):
        """
        Convert Integrated Gradients explanations to pixel importance maps.
        """
        # Convert signed attributions to scalar importance
        importance = np.abs(explanations).sum(axis=1)

        # Normalize
        importance_maps = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)

        return importance_maps

if __name__ == "__main__":
    # Load a pre-trained model (ResNet18)
    model = load_model(model_ckpt="ckpts/rn18_cifar10.ckpt")

    # Create an instance of Integrated Gradients
    ig = IntegratedGradientsXai(model)

    # Load CIFAR10 dataset
    test_dataset = CIFAR10(root='./data', train=False, download=False, transform=data_transforms)
    subset = Subset(test_dataset, list(range(10)))
    test_loader = torch.utils.data.DataLoader(subset, batch_size=2, shuffle=False)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=False)

    # Generate explanations for the test dataset
    explanations = ig.explain(test_loader)

    # Visualize the first explanation
    input_tensor, label = next(iter(test_loader))
    attributions = explanations[0]
    ig.visualize(input_tensor[0], attributions, label[0].item(), save_path='ig_output.png')

    # test image
    test_image = "./sample_explanations/cat_0.jpg"
    img = Image.open(test_image)
    img = data_transforms(img).unsqueeze(0)  # Add batch dimension
    explanation = ig.explain(img, labels=torch.tensor([3]))  # Assuming label 3 for cat
    attributions = explanation[0]
    ig.visualize(img, attributions, label[0].item(), save_path='ig_cat.png')

    # importance maps and feature rankings
    importance_maps = ig.importance_maps(explanations)
    bs = explanations.shape[0]
    im_flat = importance_maps.reshape(bs, -1)
    feature_rankings = np.argsort(im_flat)[:, ::-1]
    print(type(importance_maps), type(feature_rankings))
    print(importance_maps.shape, feature_rankings.shape)
    print(im_flat[0][feature_rankings[0][:5]])  # Should be in descending order

