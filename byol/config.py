import torch.nn as nn
from torchvision import transforms


#################################################################################
# Global Parameters and Configuration                                            #
#################################################################################

# Data Transformation Parameters
# These transformations are applied to each view of the dataset

TRANSFORMS = transforms.Compose(
    [
        transforms.RandomResizedCrop(size=28, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)

SHUFFLE = True  # Whether to shuffle the dataset

#################################################################################
# Model Architecture Parameters                                                  #
#################################################################################

# Moving Average Parameter for Momentum (used in BYOL)
TAU = 0.9

# Encoder network (converts image to representation)

ENCODER = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3, padding=1),
    nn.BatchNorm2d(16),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(16, 32, kernel_size=3, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(32 * 7 * 7, 64),
    nn.ReLU(),
)

# Projection head (projects the encoder's output to a smaller space)
PROJECTION_DIM = 32  # The dimension of the projection space
PROJECTOR = nn.Sequential(
    nn.Linear(64, PROJECTION_DIM),
    nn.ReLU(),
)

# Predictor network (used to predict the projected representations of the target network)
PREDICTOR = nn.Sequential(
    nn.Linear(PROJECTION_DIM, 32),
    nn.ReLU(),
    nn.Linear(32, PROJECTION_DIM),
)

# Fine-tuning MLP (for supervised fine-tuning after pre-training)
FINE_TUNING_MLP = nn.Sequential(
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 10),  # 10 output units for classification (MNIST)
    # nn.Softmax(dim=-1) # No softmax function needs to be added bc the crossentropy loss already applies a softmax fnction
)


#################################################################################
# Training Parameters                                                            #
#################################################################################

NUM_EPOCHS_OF_THE_UNSUPERVISED_TRAINING = 2  # Number of training epochs
PATH_OF_THE_SAVED_MODEL_PARAMETERS = (
    "../../models/trained_byol_model.pth"  # Path to save the pre-trained encoder's parameters
)

NUM_EPOCHS_OF_THE_FINE_TUNING_TRAINING = 10

PATH_OF_THE_SAVED_FINE_TUNING_PARAMETERS = "../../models/fine-tuned_model.pth"

PRETRAINING_BATCH_SIZE = 64  # Batch size for training (adjustable)

FINE_TUNING_BATCH_SIZE = 1

#################################################################################
# Testing Parameters                                                             #
#################################################################################

PATH_OF_THE_BYOL_MODEL_TO_TEST = (
    '../../models/fine-tuned_model.pth'  # Path to load the fine-tuned model for evaluation
)

# Baseline model

BASELINE_MODEL = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3, padding=1),
    nn.BatchNorm2d(16),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(16, 32, kernel_size=3, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(32 * 7 * 7, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 10),
)

PATH_OF_THE_BASELINE_MODEL = '../../models/baseline_model.pth'
