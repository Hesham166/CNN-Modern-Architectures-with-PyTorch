import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets
from matplotlib import pyplot as plt


class CIFAR10DataLoader:
    def __init__(self, batch_size=32, num_workers=0, download=True, resize=(224, 224), data_root='./data'):
        """
        Initializes the CIFAR-10 DataLoader class.

        Args:
            batch_size (int): Number of samples per batch.
            num_workers (int): Number of subprocesses to use for data loading.
            download (bool): Whether to download the dataset if not available locally.
            resize (tuple): The target image size (height, width).
            data_root (str): Directory to store/download the dataset.
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.download = download
        self.resize = resize
        self.data_root = data_root

        self.transform = transforms.Compose([
            transforms.Resize(self.resize),
            transforms.ToTensor()
        ])

        self.train_dataset = torchvision.datasets.CIFAR10(
            root=self.data_root, train=True, transform=self.transform, download=self.download
        )
        self.test_dataset = torchvision.datasets.CIFAR10(
            root=self.data_root, train=False, transform=self.transform, download=self.download
        )

    def get_train_loader(self):
        """
        Returns a DataLoader for the training dataset.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def get_test_loader(self):
        """
        Returns a DataLoader for the testing dataset.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def get_idx_to_class(self):
        """
        Returns a dict with idx to class pairs.
        """
        return {idx: label for label, idx in self.train_dataset.class_to_idx.items()}


def train_step(
    train_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
):
    """
    Performs one full training pass (epoch) over the training dataset.

    Args:
        train_loader (DataLoader): DataLoader for the training dataset.
        model (nn.Module): The neural network model.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer for updating model weights.
        device (torch.device): Device to perform computations on.

    Returns:
        tuple: Average training loss and accuracy.
    """
    model.train()
    running_loss, running_corrects = 0.0, 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_corrects += (outputs.argmax(dim=1) == labels).sum().item()

    avg_loss = running_loss / len(train_loader)
    avg_accuracy = running_corrects / len(train_loader.dataset)
    return avg_loss, avg_accuracy


def eval_step(
    test_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    device: torch.device
):
    """
    Evaluates the model on the test dataset.

    Args:
        test_loader (DataLoader): DataLoader for the test dataset.
        model (nn.Module): The neural network model.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to perform computations on.

    Returns:
        tuple: Average validation loss and accuracy.
    """
    model.eval()
    running_loss, running_corrects = 0.0, 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            running_corrects += (outputs.argmax(dim=1) == labels).sum().item()

    avg_loss = running_loss / len(test_loader)
    avg_accuracy = running_corrects / len(test_loader.dataset)
    return avg_loss, avg_accuracy


def init_cnn(module):
    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        nn.init.xavier_uniform_(module.weight)

def init_kaiming(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def layer_summary(model, X_shape):
    """
    Prints a concise summary of each layer's output shape in the model.

    Args:
        model (torch.nn.Module): The PyTorch neural network model
        X_shape (tuple): Shape of the input tensor (including batch size)
    """
    X = torch.randn(*X_shape)
    print(f"Input shape: {X_shape}")
    print("-" * 40)
    
    for name, layer in model.named_modules():
        if name == "" or list(layer.children()):  # Skip root and container modules
            continue
        try:
            X = layer(X)
            print(f"{layer.__class__.__name__:<15} output shape: {tuple(X.shape)}")
        except Exception as e:
            print(f"{layer.__class__.__name__:<15} output shape: Error - {str(e)[:30]}")
    
    print("-" * 40)