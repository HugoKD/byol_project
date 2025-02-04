from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

dataset_root = "data/raw"
print(f"Searching MNIST dataset in: {dataset_root}")
dataset_ = MNIST(root=dataset_root, train=False, download=False, transform=ToTensor())
