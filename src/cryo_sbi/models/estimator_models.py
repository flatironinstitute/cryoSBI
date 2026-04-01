from typing import Tuple
import torch
import torch.nn as nn


CLASSIFIER = {}
def add_classifier(name):
    """
    Decorator to add classifier models to the CLASSIFIER registry.
    """

    def add(class_):
        CLASSIFIER[name] = class_
        return class_

    return add


class BaseClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

    def forward(self, z):
        raise NotImplementedError("Forward method not implemented!")


@add_classifier("MLP")
class MLPClassifier(BaseClassifier):
    def __init__(
        self,
        input_dim,
        num_classes,
        num_layers=3,
        nodes_per_layer=128,
        activation=nn.ReLU,
        dropout=0.0,
    ):
        super().__init__(input_dim, num_classes)

        self.classifier = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.classifier.append(nn.Linear(input_dim, nodes_per_layer))
            else:
                self.classifier.append(nn.Linear(nodes_per_layer, nodes_per_layer))
                if dropout > 0.0:
                    self.classifier.append(nn.Dropout(dropout))
            self.classifier.append(activation())
        self.classifier.append(nn.Linear(nodes_per_layer, num_classes))
        self.classifier = nn.Sequential(*self.classifier)

    def forward(self, z, tau=1.0):
        return self.classifier(z) / tau


@add_classifier("PROTOTYPE")
class PrototypeClassifier(BaseClassifier):
    def __init__(self, input_dim, num_classes, noise_scale=0.0):
        super().__init__(input_dim, num_classes)

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.noise_scale = noise_scale
        self.prototypes = nn.Parameter(torch.randn(self.num_classes, self.input_dim))

    def forward(self, z, tau=1.0):
        if self.training and self.noise_scale > 0.0:
            prototypes = self.prototypes + torch.randn_like(self.prototypes) * self.noise_scale
        else:
            prototypes = self.prototypes

        z2 = (z**2).sum(dim=1, keepdim=True)
        p2 = (prototypes**2).sum(dim=1).unsqueeze(0)
        logits = -(z2 + p2 - 2 * z @ prototypes.T) / tau

        return logits


class ClassifierWithEmbedding(nn.Module):
    def __init__(
        self,
        embedding_net: nn.Module,
        classifier: nn.Module,
    ) -> None:
        super().__init__()
        self.classifier = classifier()
        self.embedding = embedding_net()

    def forward(self, x: torch.Tensor, tau=1.0) -> torch.Tensor:
        return self.classifier(self.embedding(x), tau=tau)

    def probs(self, x: torch.Tensor, tau=1.0) -> torch.Tensor:
        return torch.nn.functional.softmax(self.forward(x, tau=tau) / tau, dim=1)
    def logits_embedding(self, x: torch.Tensor, tau=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.embedding(x)
        logits = self.classifier(embeddings, tau=tau)
        return logits, embeddings
