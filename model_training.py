import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, v2
from torchvision import models
from torch.optim import lr_scheduler
from utils import IMG_SIZE, CIFAR10_MEAN, CIFAR10_STD

Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRAINING_CONFIG = {
    'batch_size': 128,
    'learning_rate': 1e-3,
    'num_classes': 10,  # CIFAR-10
    'epochs': 25,
    'ckpt_path': './ckpts'
}

def load_cifar10(_batch_size=TRAINING_CONFIG['batch_size']):
    """
    Loads the PyTorch train/test datasets for CIFAR-10. 
    
    Args:
        _batch_size (int): Batch size for the DataLoader.
    Returns:
            Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]: Train and test DataLoaders.
    """
    data_transforms = transforms.Compose([
        v2.Resize((IMG_SIZE, IMG_SIZE)),
        ToTensor(),
        v2.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
    ])

    # Load datasets
    train_dataset = torchvision.datasets.CIFAR10(root='data', 
                                                train=True, 
                                                transform=data_transforms,  
                                                download=True)
    test_dataset = torchvision.datasets.CIFAR10(root='data', 
                                                train=False, 
                                                transform=data_transforms,
                                                download=True)

    # Create Data Loaders
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=_batch_size,
                                               shuffle=True,
                                               num_workers=8,
                                               pin_memory=True,
                                               persistent_workers=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=_batch_size,
                                              shuffle=False)

    return train_loader, test_loader

def make_rn18(learning_rate=TRAINING_CONFIG['learning_rate'], n_classes=TRAINING_CONFIG['num_classes']):
    """
    Instantiates an untrained ResNet-18 that is configured for CIFAR-10.
    
    Args:
        learning_rate (float): Learning rate for the optimizer.
        n_classes (int): Number of classes for the classification task.
    Returns:
        Tuple[torch.nn.Module, dict]: The instantiated model and training functions.
    """
    # Initialize an untrained RESNET-18, prepare for CIFAR-10
    rn18 = models.resnet18(weights=None)
    num_ftrs = rn18.fc.in_features
    rn18.fc = nn.Linear(num_ftrs, n_classes)
    rn18 = rn18.to(Device)

    # Loss function, optimizer, LR scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer_rn18 = torch.optim.Adam(rn18.parameters(), lr=learning_rate)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_rn18, step_size=7, gamma=0.1)
    train_functions = {'loss_fn': criterion, 'optimizer': optimizer_rn18, 'lr_scheduler': exp_lr_scheduler}

    return rn18, train_functions

def train_model(_model, _Dloader, _train_fns, _epochs=TRAINING_CONFIG['epochs'], _info=False):
    """
    Trains a Pytorch model. 
    
    Args:
        _model (torch.nn.Module): The model to be trained.
        _Dloader (torch.utils.data.DataLoader): The data loader for training data.
        _train_fns (dict): A dictionary containing the loss function, optimizer, and learning rate scheduler.
        _epochs (int): The number of epochs to train the model.
        _info (bool): Whether to print training information.
    Returns:
        Tuple[torch.nn.Module, list]: The trained model and a list of training losses.
    """
    loss_fn = _train_fns['loss_fn']
    optimizer = _train_fns['optimizer']
    lr_scheduler = _train_fns['lr_scheduler']
    total_steps = len(_Dloader)
    n_samples = len(_Dloader.dataset)
    avg_loss = np.inf
    tr_losses = []

    start = time.time()  # Measure training duration
    print("Training model...")
    for e in range(_epochs):
        _model.train()
        total_loss = 0.0
        for i, (images, labels) in enumerate(_Dloader):
            images = images.to(Device)
            labels = labels.to(Device)

            # Forward pass
            outputs = _model(images)
            loss = loss_fn(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            current_lr = lr_scheduler.get_last_lr()[0]

            sys.stderr.write(f"\rEpochs {e+1:02d}/{_epochs:02d} | Steps {i}/{total_steps} | Epoch Loss: {total_loss:<6.4f} | Total Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")
            sys.stderr.flush()

        # Update learning rate
        lr_scheduler.step()

        avg_loss = total_loss / n_samples
        tr_losses.append(avg_loss)

    time_elapsed = time.time() - start
    print(f"\nFinished training in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")

    # Save the model weights
    ckpt_path = TRAINING_CONFIG['ckpt_path']
    os.makedirs(ckpt_path, exist_ok=True)
    torch.save(_model.state_dict(), f"{ckpt_path}/rn18_cifar10.ckpt")

    return _model, tr_losses

def plot_training_progress(_losses):
    """
    Plots training progress in terms of loss vs. epochs.
    
    Args:
        _losses (list): List of training losses.
    """
    plt.figure(figsize=(10,4), dpi=72)
    plt.gca().set(xlabel='Epoch', ylabel='loss', title='Training loss')
    plt.plot(_losses)
    plt.show()

def test_model(_model, _test_loader):
    """
    Evaluates a trained model on the test dataset.
    
    Args:
        _model (torch.nn.Module): The trained model to be evaluated.
        _test_loader (torch.utils.data.DataLoader): The data loader for the test dataset.
    Returns:
        torch.Tensor: A tensor containing the model's predictions for each test sample.
    """
    num_samples = len(_test_loader.dataset)

    # Test the model
    all_predictions = []
    _model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in _test_loader:
            images = images.to(Device)
            labels = labels.to(Device)
            outputs = _model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.append(predicted)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        all_predictions = torch.cat(all_predictions)

        print(f"\nTest Accuracy of the model on the {num_samples} test images: {100 * correct / total} %")

    return all_predictions

def save_correct_predictions(_preds, _ts_dataset, _save_path=TRAINING_CONFIG['ckpt_path']):
    """
    Helper function to save the correct model predictions indices.
    
    Args:
        _preds (torch.Tensor): The model predictions for the test dataset.
        _ts_dataset (torch.utils.data.Dataset): The test dataset.
        _save_path (str): The path to save the correct predictions indices.
    """
    os.makedirs(_save_path, exist_ok=True)
    print(f"\nSaving Correct predictions to '{_save_path}'...")
    predictions = _preds.detach().cpu().numpy()
    targets = np.array([_[1] for _ in _ts_dataset])
    correct_indices = np.where(predictions == targets)[0]
    print(f"# correct={len(correct_indices)}")
    print('First 5:')
    class_labels = _ts_dataset.classes
    for i in range(5):
        idx = correct_indices[i]
        target = _ts_dataset[idx][1]
        print(f"idx: {idx}, class: '{class_labels[target]}'")
    np.save(f"{_save_path}/correct_preds_indices.npy", correct_indices)


if __name__=='__main__':
    # Are we using a GPU or CPU?
    print(Device)

    # Load CIFAR-10
    tr_dloader, ts_loader = load_cifar10()

    # Create a new ResNet-18 model configured for CIFAR-10
    rn18, train_fns = make_rn18()

    # Training
    rn18, losses = train_model(_model=rn18,
                               _Dloader=tr_dloader,
                               _train_fns=train_fns,
                               _info=True)

    # Sanity Check
    plot_training_progress(losses)

    # Model Evaluation
    correct_preds = test_model(rn18, ts_loader)

    # Save the correct predictions indices for generating model explanations.
    save_correct_predictions(correct_preds, ts_loader.dataset)