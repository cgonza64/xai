import torch
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, v2
from torchvision.models import resnet18
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Subset
import numpy as np

# CIFAR10 dataset statistics
NUM_CLASSES = 10
IMG_SIZE = 224
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2023, 0.1994, 0.2010]
BATCH_SIZE = 128

data_transforms = transforms.Compose([
    v2.Resize((IMG_SIZE, IMG_SIZE)),
    ToTensor(),
    v2.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
])

lime_transforms = transforms.Compose([
    v2.Resize((IMG_SIZE, IMG_SIZE))
])

def tensor2image(_in_tensor, _mean=CIFAR10_MEAN, _std=CIFAR10_STD):
    """ Converts an input tensor back to a numpy image. I.e, reverses the data transforms defined above. """
    inp = _in_tensor.squeeze(0).numpy().transpose((1, 2, 0))
    inp = _std * inp + _mean
    return np.clip(inp, 0, 1)

def load_data_samples(correct_indices_pth="./ckpts/correct_preds_indices.npy",
                      batch_size=BATCH_SIZE,
                      transforms=data_transforms,
                      data_only=False
                      ):
    """
    Helper function to load the correctly predicted samples from the CIFAR-10 dataset.
    The correct predictions are based on the pre-trained ResNet-18 model whose weights are 
    saved under the ckpts directory.
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
    """ Helper function to load a pre-trained ResNet-18 model. """
    model = resnet18(weights=None, num_classes=NUM_CLASSES)  # Assuming CIFAR-10 dataset with 10 classes
    checkpoint = torch.load(model_ckpt, map_location="cpu")
    model.load_state_dict(checkpoint)
    return model