import torch.nn as nn
from torch import no_grad


class Encoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        return self.encoder(x)


class Projector(nn.Module):
    def __init__(self, projector):
        super().__init__()
        self.projector = projector

    def forward(self, x):
        return self.projector(x)


class Predictor(nn.Module):
    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor

    def forward(self, x):
        return self.predictor(x)


class BootstrapYourOwnLatent(nn.Module):
    """
    Implements the BYOL (Bootstrap Your Own Latent) self-supervised learning framework.

    BYOL leverages two networks (online and target) to learn representations without the need
    for negative samples. The online network learns to predict the target network's representations,
    while the target network's parameters are updated as an exponential moving average of the
    online network's parameters.

    ------------------------------------------------------------------------
    Attributes:
    ------------------------------------------------------------------------
        tau (float):
            The momentum coefficient used for updating the target network's parameters.
        online_encoder (nn.Module):
            Encoder for the online network, responsible for extracting features from input images.
        online_projector (nn.Module):
            Projector for the online network, mapping encoder outputs to a latent space.
        online_predictor (nn.Module):
            Predictor for the online network, predicting the target network's latent representations.
        target_encoder (nn.Module):
            Encoder for the target network, fixed during backpropagation.
        target_projector (nn.Module):
            Projector for the target network, fixed during backpropagation.

    ------------------------------------------------------------------------
    Methods:
    ------------------------------------------------------------------------
        update_the_moving_average_for_the_encoder():
            Updates the target encoder's parameters as an exponential moving average of the online
            encoder's parameters.

        update_the_moving_average_for_the_projector():
            Updates the target projector's parameters as an exponential moving average of the online
            projector's parameters.

        compute_loss(x, y):
            Computes the cosine similarity-based loss between normalized vectors `x` and `y`.

        forward(v1, v2):
            Computes the BYOL loss given two augmented views (`v1` and `v2`) of the same image.
    """

    def __init__(self, encoder, projector, predictor, tau):
        super().__init__()

        self.tau = tau

        self.online_encoder = encoder
        self.online_projector = projector
        self.online_predictor = predictor

        self.target_encoder = encoder
        self.target_projector = projector

    def update_the_moving_average_for_the_encoder(self):
        with no_grad():
            for theta, ksi in zip(
                self.online_encoder.parameters(), self.target_encoder.parameters()
            ):
                new_ksi = self.tau * ksi + (1 - self.tau) * theta
                ksi.data = new_ksi

    def update_the_moving_average_for_the_projector(self):
        with no_grad():
            for theta, ksi in zip(
                self.online_projector.parameters(), self.target_projector.parameters()
            ):
                new_ksi = self.tau * ksi + (1 - self.tau) * theta
                ksi.data = new_ksi

    @staticmethod
    def compute_loss(x, y):
        x = nn.functional.normalize(x, dim=-1)
        y = nn.functional.normalize(y, dim=-1)
        return 2 - 2 * (x * y).sum(dim=-1)

    def forward(self, v1, v2):
        online_prediction1 = self.online_predictor(self.online_projector(self.online_encoder(v1)))
        online_prediction2 = self.online_predictor(self.online_projector(self.online_encoder(v2)))

        with no_grad():
            target_prediction1 = self.target_projector(self.target_encoder(v1))
            target_prediction2 = self.target_projector(self.target_encoder(v2))

        loss = (
            self.compute_loss(online_prediction1, target_prediction1).mean()
            + self.compute_loss(online_prediction2, target_prediction2).mean()
        )
        return loss


class FineTunedBootstrapYourOwnLatent(nn.Module):
    """
    Fine-tuning module for the BYOL (Bootstrap Your Own Latent) framework.

    This class is designed to use a pre-trained BYOL encoder and an additional
    Multi-Layer Perceptron (MLP) for supervised tasks, such as classification.

    ------------------------------------------------------------------------
    Attributes:
    ------------------------------------------------------------------------
        mlp (nn.Module):
            The fine-tuning MLP responsible for mapping encoded features to the desired output space.
        encoder (nn.Module):
            The pre-trained encoder module that extracts features from input data.

    ------------------------------------------------------------------------
    Methods:
    ------------------------------------------------------------------------
        forward(x):
            Computes the forward pass through the encoder and the fine-tuning MLP.

    """

    def __init__(self, encoder, fine_tuning_mlp):
        super().__init__()
        self.mlp = fine_tuning_mlp
        self.encoder = encoder

    def forward(self, x):
        # with no_grad():
        x = self.encoder(x)
        x = self.mlp(x)
        return x


class Baseline(nn.Module):
    """
    Baseline module which is used to compare the performances of fine-tuned BYOL

    ------------------------------------------------------------------------
    Attributes:
    ------------------------------------------------------------------------
        baseline (nn.Sequential):
            The baseline architecture.

    ------------------------------------------------------------------------
    Methods:
    ------------------------------------------------------------------------
        forward(x):
            Computes the forward pass through the baseline.

    """

    def __init__(self, baseline):
        super().__init__()
        self.baseline = baseline

    def forward(self, x):
        x = self.baseline(x)
        return x