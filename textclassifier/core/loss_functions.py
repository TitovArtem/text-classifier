from abc import abstractmethod
import numpy as np


class LossFunction(object):

    @abstractmethod
    def value(self, x, y, w):
        """
        Return value of the loss function.
        :param x: numpy array, features
        :param y: numpy array, results
        :param w: numpy array, coefficients of linear hyperplane
        """

    @abstractmethod
    def derivative(self, x, y, w):
        """ Return the derivative of the loss function. """


class LogLossFunction(LossFunction):

    def derivative(self, x, y, w):
        t = y * (1.0 - 1.0 / (1. + np.e ** (-y * x.dot(w)))) / y.size
        return -x.T.dot(t)

    def value(self, x, y, w):
        return np.log2(1 + np.e ** (-y * x * w))
