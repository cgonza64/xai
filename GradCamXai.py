from XaiMethodBase import XaiMethodBase
import matplotlib.pyplot as plt
from PIL import Image
from utils import data_transforms, load_model, tensor2image
from tqdm import tqdm
import numpy as np

# Torch
import torch
import cv2
from torchvision.datasets import CIFAR10
from torch.utils.data import Subset

# GradCAM
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

class GradCamXai(XaiMethodBase):
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping) implementation for generating visual explanations.
    """

    def __init__(self, model, target_layers):
        super().__init__(model)
        self.target_layers = target_layers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def explain(self, input_data, labels=None):
        """
        Generate Grad-CAM explanations for the given input data.

        Args:
            input_data: The input data for which explanations are to be generated.

        Returns:
            Grad-CAM heatmaps for the input data.
        """
        # Generate CAM explanations
        explanations = []
        self.model.to(self.device)
        self.model.eval()
        cam = GradCAM(model=self.model, target_layers=self.target_layers)
        if isinstance(input_data, torch.utils.data.DataLoader):
            for imgs, labels in tqdm(input_data, desc="Generating Grad-CAM explanations"):
                imgs = imgs.to(self.device)
                grayscale_cams = cam(input_tensor=imgs, targets=[ClassifierOutputTarget(t) for t in labels])
                explanations.append(grayscale_cams)
        elif isinstance(input_data, torch.Tensor):
            assert labels is not None, "Labels must be provided when input_data is a Tensor."
            for img in input_data:
                img = img.to(self.device)
                grayscale_cams = cam(input_tensor=img.unsqueeze(0), targets=[ClassifierOutputTarget(t.item()) for t in labels])
                explanations.append(grayscale_cams)
        else:
            raise ValueError("Input data must be a DataLoader or a Tensor.")

        return np.concatenate(explanations, axis=0)

    def visualize(self, input_tensor, grayscale_cam, target_class, show=True, save_path=None):
        """
        Visualize the Grad-CAM heatmap on the input image.

        Args:
            input_tensor: The original input image tensor.
            grayscale_cam: The Grad-CAM heatmap for the input image.
            target_class: The class label for which the explanation is generated.
            show: Whether to display the visualization.
            save_path: Path to save the visualization image. If None, the image is not saved.
        """
        # Visualize
        inp = tensor2image(input_tensor)
        visualization = show_cam_on_image(inp, grayscale_cam, use_rgb=True)
        _, ax = plt.subplots(1, 1)
        ax.imshow(visualization)
        ax.set_title(f'GradCAM - Target Class: {target_class}')
        ax.axis("off")

        # Save image to file
        if save_path is not None:
            cv2.imwrite(save_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

        if show:
            plt.show()

        plt.close()

    def importance_maps(self, explanations):
        """
        Convert Grad-CAM heatmaps to pixel importance maps.

        Args:
            explanations: A numpy array of Grad-CAM heatmaps for each input image.
        Returns:
            A numpy array of pixel importance maps for each input image.
        """
        return explanations  # GradCAM already produces importance mappings that are normalized in [0, 1]


if __name__ == "__main__":
    # Load a pre-trained model (ResNet18)
    model = load_model(model_ckpt="ckpts/rn18_cifar10.ckpt")

    # Define the target layer for Grad-CAM
    target_layers = [model.layer4[-1]]  # Last layer of the last block in ResNet18

    # Create an instance of GradCam
    grad_cam = GradCamXai(model, target_layers)

    # Load CIFAR10 dataset
    transform = data_transforms
    test_dataset = CIFAR10(root='./data', train=False, download=False, transform=transform)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False)
    subset = Subset(test_dataset, list(range(10)))
    test_loader = torch.utils.data.DataLoader(subset, batch_size=128, shuffle=False)

    # Generate explanations for the test dataset
    explanations = grad_cam.explain(test_loader)

    # # Visualize the first explanation
    input_tensor, label = next(iter(test_loader))
    grayscale_cam = explanations[0]
    grad_cam.visualize(input_tensor[0], grayscale_cam, label[0].item(), save_path='gradcam_output.png')

    # test image
    test_image = "C:/Users/cesar/Su26_Advanced_AppliedML/research_project/custom_animals_dataset/cat/cat_0.jpg"
    img = Image.open(test_image)
    img = data_transforms(img).unsqueeze(0)  # Add batch dimension
    explanation = grad_cam.explain(img, labels=torch.tensor([3]))  # Assuming label 3 for cat
    grayscale_cam = explanation[0]
    grad_cam.visualize(img, grayscale_cam, label[0].item(), save_path='gradcam_cat.png')

    # importance maps and feature rankings
    importance_maps = grad_cam.importance_maps(explanations)
    bs = explanations.shape[0]
    im_flat = importance_maps.reshape(bs, -1)
    feature_rankings = np.argsort(im_flat)[:, ::-1]
    print(type(importance_maps), type(feature_rankings))
    print(importance_maps.shape, feature_rankings.shape)
    print(im_flat[0][feature_rankings[0][:5]])  # Should be in descending order

