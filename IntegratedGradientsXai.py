from XaiMethodBase import XaiMethodBase
import matplotlib.pyplot as plt
from PIL import Image
from utils import data_transforms, load_model, tensor2image
from tqdm import tqdm
import numpy as np

# Torch
import torch
import numpy as np
import cv2
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18

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


        # Generate IG explanations
        explanations, deltas = [], []
        self.model.to(self.device)
        self.model.eval()
        ig = IntegratedGradients(model)

        # # Load a batch of inputs
        # input_tensor, labels = next(iter(dloader))
        # input_tensor = input_tensor.to(Device)

        # # Baseline input
        # baseline = torch.zeros_like(input_tensor).to(Device)

        if isinstance(input_data, torch.utils.data.DataLoader):
            # Baseline input
            images, _ = next(iter(input_data))
            baseline = torch.zeros_like(images).to(self.device)  
            for imgs, targets in tqdm(input_data, desc="Generating IG explanations"):
                imgs = imgs.to(self.device)
                with torch.no_grad():
                    logits = self.model(imgs)
                    pred_classes = logits.argmax(dim=1)   # shape [B]

                attributions, delta = ig.attribute(
                    inputs=imgs,
                    baselines=baseline,
                    target=pred_classes,
                    n_steps=50,
                    return_convergence_delta=True
                )
                explanations.append(attributions)
                deltas.append(delta)
        elif isinstance(input_data, torch.Tensor):
            assert labels is not None, "Labels must be provided when input_data is a Tensor."
            for img in input_data:
                img = img.to(self.device)
                # grayscale_cams = cam(input_tensor=img.unsqueeze(0), targets=[ClassifierOutputTarget(t.item()) for t in labels])
                # explanations.append(grayscale_cams)
        else:
            raise ValueError("Input data must be a DataLoader or a Tensor.")

        explanations = torch.cat(explanations, dim=0)
        deltas = torch.cat(deltas, dim=0)
        print("Attributions shape:", explanations.shape)
        print("Delta shape:", deltas.shape)
        print("Mean convergence delta:", deltas.abs().mean().item())
        return explanations.detach().cpu().numpy()

    def visualize(self, input_tensor, grayscale_cam, target_class, show=True, save_path=None):
        """
        Visualize the Integrated Gradients heatmap on the input image.
        """
        # Visualize
        inp = tensor2image(input_tensor)
        visualization = show_cam_on_image(inp, grayscale_cam, use_rgb=True)
        _, ax = plt.subplots(1, 1)
        ax.imshow(visualization)
        ax.set_title(f'GradCAM Output - Target Class: {target_class}')
        if show:
            plt.show()

        # Save image to file
        if save_path is not None:
            cv2.imwrite(save_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

    def importance_maps(self, explanations):
        """
        Convert Integrated Gradients explanations to pixel importance maps.
        """
        # bs = explanations.shape[0]
        # exp_flat = explanations.reshape(bs, -1)
        # importance_maps = np.argsort(exp_flat)[:, ::-1]
        # return importance_maps, exp_flat
        return explanations  # GradCAM already produces importance mappings that are normalized in [0, 1]


if __name__ == "__main__":
    # Load a pre-trained model (ResNet18)
    model = load_model(model_ckpt="ckpts/rn18_cifar10.ckpt")

    # Create an instance of GradCam
    ig = IntegratedGradientsXai(model)

    # Load CIFAR10 dataset
    transform = data_transforms
    test_dataset = CIFAR10(root='./data', train=False, download=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=False)

    # Generate explanations for the test dataset
    explanations = ig.explain(test_loader)

    # # # Visualize the first explanation
    # input_tensor, label = next(iter(test_loader))
    # grayscale_cam = explanations[0]
    # grad_cam.visualize(input_tensor[0], grayscale_cam, label[0].item(), save_path='gradcam_output.png')

    # # test image
    # test_image = "C:/Users/cesar/Su26_Advanced_AppliedML/research_project/custom_animals_dataset/cat/cat_0.jpg"
    # img = Image.open(test_image)
    # img = data_transforms(img).unsqueeze(0)  # Add batch dimension
    # explanation = grad_cam.explain(img, labels=torch.tensor([3]))  # Assuming label 3 for cat
    # grayscale_cam = explanation[0]
    # grad_cam.visualize(img, grayscale_cam, label[0].item(), save_path='gradcam_cat.png')

    # # importance maps and feature rankings
    # importance_maps = grad_cam.importance_maps(explanations)
    # bs = explanations.shape[0]
    # im_flat = importance_maps.reshape(bs, -1)
    # feature_rankings = np.argsort(im_flat)[:, ::-1]
    # print(type(importance_maps), type(feature_rankings))
    # print(importance_maps.shape, feature_rankings.shape)
    # print(im_flat[0][feature_rankings[0][:5]])  # Should be in descending order

