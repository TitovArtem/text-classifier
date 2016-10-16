from abc import abstractmethod
import numpy as np

from textclassifier.core.base import SupervisedModel, ClassifierMixin
from textclassifier.core.gradient_methods import GradientDescent


class LinearModel(SupervisedModel):

    def __init__(self, optimizer=GradientDescent()):
        super().__init__()
        self.w = None
        self.optimizer = optimizer

    def train(self, x, y):
        self.w = self.optimizer.transform(x, y, self.w)

    @abstractmethod
    def predict(self, x):
        if self.w is None:
            raise ValueError("The model is not trained.")


class LinearClassifier(LinearModel, ClassifierMixin):

    def predict(self, x):
        super().predict(x)
        return np.sign(x.dot(self.w))

    def predict_probability(self, x):
        if self.w is None:
            raise ValueError("The model is not trained.")
        return 1.0 / (1.0 + np.e ** (-x.dot(self.w)))
