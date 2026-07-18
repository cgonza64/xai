import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, v2
import numpy as np

# CIFAR10 dataset statistics
NUM_CLASSES = 10
IMG_SIZE = 224
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2023, 0.1994, 0.2010]

data_transforms = transforms.Compose([
    v2.Resize((IMG_SIZE, IMG_SIZE)),
    ToTensor(),
    v2.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
])

def tensor2numpy(_in_tensor, _mean=CIFAR10_MEAN, _std=CIFAR10_STD):
    """ Converts an input tensor back to a numpy image. I.e, reverses the data transforms defined above. """
    inp = _in_tensor.squeeze(0).numpy().transpose((1, 2, 0))
    inp = _std * inp + _mean
    return np.clip(inp, 0, 1)