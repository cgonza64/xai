import torch
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, v2
from torchvision.models import resnet18
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Subset
import numpy as np

# CIFAR10 dataset statistics/configuration
NUM_CLASSES = 10
IMG_SIZE = 224
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2023, 0.1994, 0.2010]
BATCH_SIZE = 128

# Data transformations for the CIFAR-10 dataset
data_transforms = transforms.Compose([
    v2.Resize((IMG_SIZE, IMG_SIZE)),
    ToTensor(),
    v2.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
])

# LIME requires the input image to be in the range [0, 1] and not normalized, so we define a separate transformation for LIME.
lime_transforms = transforms.Compose([
    v2.Resize((IMG_SIZE, IMG_SIZE))
])

def tensor2image(_in_tensor, _mean=CIFAR10_MEAN, _std=CIFAR10_STD):
    """
    Converts an input tensor back to a numpy image. I.e, reverses the data transforms defined above.

    Args:
        _in_tensor (torch.Tensor): Input tensor of shape (1, channels, height, width) to be converted to a numpy image.
        _mean (list): List of mean values used for normalization.
        _std (list): List of standard deviation values used for normalization.
    Returns:
        np.ndarray: Numpy array of shape (height, width, channels) representing the image
    """
    inp = _in_tensor.squeeze(0).numpy().transpose((1, 2, 0))
    inp = _std * inp + _mean
    return np.clip(inp, 0, 1)

def load_data_samples(correct_indices_pth="./ckpts/correct_preds_indices.npy",
                      batch_size=BATCH_SIZE,
                      transforms=data_transforms,
                      data_only=False):
    """
    Helper function to load the correctly predicted samples from the CIFAR-10 dataset.
    The correct predictions are based on the pre-trained ResNet-18 model whose weights are 
    saved under the ckpts directory.

    Args:
        correct_indices_pth (str): Path to the numpy file containing the indices of correctly predicted samples.
        batch_size (int): Batch size for the DataLoader.
        transforms (torchvision.transforms.Compose): Transformations to be applied to the dataset.
        data_only (bool): If True, only return the image samples as a numpy array (used for LIME). If False, return a DataLoader for the correct samples.
    Returns:
        If data_only is True:
            np.ndarray: Numpy array of shape (num_samples, channels, height, width) containing the image samples.
        If data_only is False:
            torch.utils.data.DataLoader: DataLoader for the correct samples.
    """
    test_dataset = CIFAR10(root='data', train=False, transform=transforms)
    correct_indices = np.load(correct_indices_pth)
    correct_samples = Subset(test_dataset, correct_indices)

    # Only return the image samples as a numpy array (used for LIME)
    if data_only:
        images = [correct_samples[_][0] for _ in range(len(correct_samples))]
        return np.array(images)

    dloader = DataLoader(dataset=correct_samples,
                         batch_size=batch_size,
                         shuffle=False)
    return dloader

def load_model(model_ckpt="ckpts/rn18_cifar10.ckpt"):
    """
    Helper function to load a pre-trained ResNet-18 model.

    Args:
        model_ckpt (str): Path to the model checkpoint file.
    Returns:
        torch.nn.Module: Pre-trained ResNet-18 model."""
    model = resnet18(weights=None, num_classes=NUM_CLASSES)  # Assuming CIFAR-10 dataset with 10 classes
    checkpoint = torch.load(model_ckpt, map_location="cpu")
    model.load_state_dict(checkpoint)
    return model