import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


from byol.dataset import BYOLDataset
from byol.modeling.models import BootstrapYourOwnLatent, Encoder, Projector, Predictor, Baseline

from byol.config import (
    ENCODER,
    PROJECTOR,
    PREDICTOR,
    NUM_EPOCHS_OF_THE_UNSUPERVISED_TRAINING,
    TRANSFORMS,
    PRETRAINING_BATCH_SIZE,
    SHUFFLE,
    TAU,
    PATH_OF_THE_SAVED_MODEL_PARAMETERS,
    BASELINE_MODEL,
    PATH_OF_THE_BASELINE_MODEL
)
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torch import device, cuda, save, max, load, equal
import torch.nn as nn
from torchvision.transforms.v2 import ToTensor

from byol.modeling.models import FineTunedBootstrapYourOwnLatent, Encoder
from byol.config import (
    ENCODER,
    NUM_EPOCHS_OF_THE_FINE_TUNING_TRAINING,
    SHUFFLE,
    PATH_OF_THE_SAVED_MODEL_PARAMETERS,
    FINE_TUNING_MLP,
    PATH_OF_THE_SAVED_FINE_TUNING_PARAMETERS,
    FINE_TUNING_BATCH_SIZE,
)
import torch.optim as optim

print(PATH_OF_THE_SAVED_MODEL_PARAMETERS)

class BYOLPreTrainer:
    def __init__(self, train_dataset, test_dataset,model, batch_size, shuffle, number_of_epochs, save_path):

        self.device = device("cuda" if cuda.is_available() else "cpu")
        self.model = model
        self.train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle)
        self.test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=shuffle)
        self.number_of_epochs = number_of_epochs
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=0.0003
        )  # TODO: add the possibility to use SGD
        self.save_path = save_path

    def __call__(self):
        self.model.train()
        for epoch in range(self.number_of_epochs):
            total_loss = 0
            for index, (view1, view2, label) in enumerate(self.train_dataloader):
                view1, view2 = view1.to(self.device), view2.to(self.device)
                self.optimizer.zero_grad()
                loss = self.model(view1, view2)

                loss.backward()
                self.optimizer.step()
                self.model.update_the_moving_average_for_the_encoder()
                self.model.update_the_moving_average_for_the_projector()

                total_loss += loss.item()
                if index % 100 == 0:
                    print(
                        f"Epoch [{epoch + 1}/{NUM_EPOCHS_OF_THE_UNSUPERVISED_TRAINING}], Iter [{index + 1}/{len(self.train_dataloader)}], Loss: {loss.item():.4f}"
                    )
            print(
                f"Epoch {epoch + 1} completed. Average loss: {total_loss / len(self.train_dataloader):.4f}"
            )
        print("Training ended !")

        save(self.model.online_encoder.state_dict(), self.save_path)


class SupervisedTrainer:
    def __init__(self, dataset, model, batch_size, shuffle, number_of_epochs, loss_fn, save_path):
        self.device = device("cuda" if cuda.is_available() else "cpu")
        self.model = model
        self.train_dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
        self.number_of_epochs = number_of_epochs
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0003)
        self.save_path = save_path
        self.criterion = loss_fn

    def __call__(self):
        for epoch in range(self.number_of_epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in self.train_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            epoch_loss = running_loss / len(self.train_dataloader)
            epoch_acc = correct / total * 100

            print(
                f"Epoch [{epoch + 1}/{self.number_of_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%"
            )

        save(self.model.state_dict(), self.save_path)


if __name__ == "__main__":
    if len(sys.argv) >= 1:
        match sys.argv[1]:
            case "pre_train":
                encoder = Encoder(ENCODER)
                projector = Projector(PROJECTOR)
                predictor = Predictor(PREDICTOR)
                model = BootstrapYourOwnLatent(encoder, projector, predictor, TAU)
                model = model.to(device("cuda" if cuda.is_available() else "cpu"))

                raw_dataset = MNIST(root="data/raw", train=True, download=True)
                train_transformed_dataset = BYOLDataset(raw_dataset, TRANSFORMS)

                print("Now pre-training the BYOL model...")
                trainer = BYOLPreTrainer(
                    train_dataset=train_transformed_dataset,
                    model=model,
                    batch_size=PRETRAINING_BATCH_SIZE,
                    shuffle=SHUFFLE,
                    number_of_epochs=NUM_EPOCHS_OF_THE_UNSUPERVISED_TRAINING,
                    save_path=PATH_OF_THE_SAVED_MODEL_PARAMETERS,
                )
                trainer()

            case "fine_tune":
                encoder = Encoder(ENCODER)
                encoder.load_state_dict(
                    load(PATH_OF_THE_SAVED_MODEL_PARAMETERS, weights_only=True)
                )

                model = FineTunedBootstrapYourOwnLatent(encoder, FINE_TUNING_MLP)
                model = model.to(device("cuda" if cuda.is_available() else "cpu"))
                dataset = MNIST(root="data/raw", train=True, download=True, transform=ToTensor())

                subset_indices = torch.randperm(len(dataset))[:100]
                subset_dataset = Subset(dataset, subset_indices)

                print("Now fine-tuning the BYOL model...")
                trainer = SupervisedTrainer(
                    train_dataset=subset_dataset,
                    test_dataset=subset_dataset,
                    model=model,
                    batch_size=FINE_TUNING_BATCH_SIZE,
                    shuffle=SHUFFLE,
                    number_of_epochs=NUM_EPOCHS_OF_THE_FINE_TUNING_TRAINING,
                    loss_fn=nn.CrossEntropyLoss(),
                    save_path=PATH_OF_THE_SAVED_FINE_TUNING_PARAMETERS,
                )
                trainer()

            case "baseline":
                model = Baseline(BASELINE_MODEL)
                model = model.to(device("cuda" if cuda.is_available() else "cpu"))
                dataset = MNIST(root="data/raw", train=True, download=True, transform=ToTensor())

                subset_indices = torch.randperm(len(dataset))[:100]
                subset_dataset = Subset(dataset, subset_indices)

                print("Now training the baseline model...")
                baseline_trainer = SupervisedTrainer(
                    dataset=subset_dataset,
                    model=model,
                    batch_size=FINE_TUNING_BATCH_SIZE,
                    shuffle=SHUFFLE,
                    number_of_epochs=NUM_EPOCHS_OF_THE_FINE_TUNING_TRAINING,
                    loss_fn=nn.CrossEntropyLoss(),
                    save_path=PATH_OF_THE_BASELINE_MODEL,
                )
                baseline_trainer()

            case _:
                raise Exception(
                    "Please enter a valid argument. You can either pre-train the BYOL model, fine-tune it or train the baseline model. Supported arguments : 'pre_train', 'fine_tune' and 'baseline'"
                )
