import sys
import os
from pathlib import Path
print(os.getcwd()) #/home/cadet/PycharmProjects/byol_project/byol/modeling
from models import FineTunedBootstrapYourOwnLatent, Encoder, Baseline
from torch import load, no_grad, device, cuda, max
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (
    ENCODER,
    PRETRAINING_BATCH_SIZE,
    SHUFFLE,
    PATH_OF_THE_BYOL_MODEL_TO_TEST,
    FINE_TUNING_MLP, BASELINE_MODEL, PATH_OF_THE_BASELINE_MODEL
)
print(PATH_OF_THE_BASELINE_MODEL)
class Predictor:
    def __init__(self, dataset, model, path_of_the_model_parameters, end_log):
        self.device = device("cuda" if cuda.is_available() else "cpu")
        self.model = model
        self.test_dataloader = DataLoader(dataset=dataset)
        self.end_log = end_log

        self.model.load_state_dict(load(path_of_the_model_parameters, weights_only=True))

    def __call__(self):
        self.model.eval()
        self.model.to(self.device)
        total = 0
        correct = 0
        with no_grad():
            for index, (view, labels) in enumerate(self.test_dataloader):
                view = view.to(self.device)
                labels = labels.to(self.device)

                output = self.model(view)

                _, predicted = max(output, 1)

                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        accuracy = 100 * correct / total
        print(f"Accuracy of the {self.end_log}: {accuracy:.2f}%")


if __name__ == '__main__':
    if len(sys.argv)>=1:
        dataset_ = MNIST(root="data/raw", train=False, download=False, transform=ToTensor())
        print(os.getcwd())
        print(dataset_)
        match sys.argv[1]:
            case 'byol':
                encoder = Encoder(ENCODER)
                model_ = FineTunedBootstrapYourOwnLatent(encoder, FINE_TUNING_MLP)
                predictor = Predictor(
                    dataset=dataset_,
                    model=model_,
                    path_of_the_model_parameters=PATH_OF_THE_BYOL_MODEL_TO_TEST,
                    end_log='Boostrap Your Own Latent model'
                )

                predictor()

            case 'baseline':
                model_ = Baseline(BASELINE_MODEL)
                predictor = Predictor(
                    dataset=dataset_,
                    model=model_,
                    path_of_the_model_parameters=PATH_OF_THE_BASELINE_MODEL,
                    end_log='Baseline model'
                )

                predictor()

            case _:
                raise Exception(
                    "Please enter a valid argument. You can either test the BYOL model or the baseline model. Supported arguments : 'byol' and 'baseline'"
                )
