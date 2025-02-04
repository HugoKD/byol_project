from torch.utils.data import DataLoader, Dataset


class BYOLDataset(Dataset):
    """
    Custom dataset class for generating multiple augmented views of the same input image,
    specifically designed for training the BYOL (Bootstrap Your Own Latent) framework.

    ------------------------------------------------------------------------
    Attributes:
    ------------------------------------------------------------------------
        dataset (Dataset):
            The base dataset containing the raw data (e.g., MNIST, CIFAR-10).
        transform (callable):
            The transformation function or composition of transformations
            applied to the input images to generate augmented views.

    ------------------------------------------------------------------------
    Methods:
    ------------------------------------------------------------------------
        __len__():
            Returns the size of the dataset.

        __getitem__(idx):
            Retrieves an item from the dataset and generates two augmented views of the same image.
    """

    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        view1 = self.transform(image)  # We generate two views for the same image
        view2 = self.transform(image)
        return view1, view2, label
