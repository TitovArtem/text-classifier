""" Optimizers are based on gradient methods. """

from abc import abstractmethod
import numpy as np

from textclassifier.core.base import BaseEstimator, AttributeTypeError, \
    Optimizer
from textclassifier.core.loss_functions import LogLossFunction, LossFunction


class GradientMethod(Optimizer):
    """ Abstract class for all gradient methods. """

    def __init__(self, loss_func=LogLossFunction(), max_iter=int(1e3),
                 eps=0.0001, step=1e-3):
        self.loss_func = loss_func
        self.max_iter = max_iter
        self.eps = eps
        self.step = step
        self._cur_iter = 0
        self._validate_params()

    def _validate_params(self):
        if type(self.max_iter) is not int:
            raise AttributeTypeError(self.max_iter.__name__, int.__name__)

        if self.max_iter <= 0:
            raise ValueError("Attribute 'mat_iter' must have "
                             "value more then 0.")

        if not isinstance(self.eps, (float, int)):
            raise AttributeTypeError(self.eps.__name__, float.__name__)

        if self.eps < 0.0:
            raise ValueError("Attribute 'eps' must be positive.")

        if not isinstance(self.step, (float, int)):
            raise AttributeTypeError(self.step.__name__, float.__name__)

        if self.step < 0.0:
            raise ValueError("Attribute 'step' must be positive.")

        if not isinstance(self.loss_func, LossFunction):
            raise ValueError("Attribute %s must be instance of LossFunction."
                             % self.loss_func.__name__)

    def set_params(self, **params):
        super().set_params(**params)
        self._validate_params()

    @property
    def count_iterations(self):
        """ Return the count of iterations for the last optimization. """
        return self._cur_iter

    @abstractmethod
    def transform(self, x, y, w=None):
        """ Return numpy array of coefficients. """


class GradientDescent(GradientMethod):
    """ The simple gradient descent. """

    def __init__(self, verbose=False, *args, **kwargs):
        self.verbose = verbose
        super().__init__(*args, **kwargs)

    def _validate_params(self):
        super()._validate_params()
        if type(self.verbose) is not bool:
            raise AttributeTypeError(self.verbose.__name__, bool.__name__)

    def transform(self, x, y, w=None):
        self._validate_params()
        if x.shape[0] != y.shape[0]:
            raise AttributeError("The given data sets have not "
                                 "the same dimensions.")

        if w is None:
            w = np.zeros(x.shape[1])

        if not self.loss_func:
            raise ValueError("The given function for calculating a derivative "
                             "value is not defined.")

        self._cur_iter = 0
        for self._cur_iter in range(self.max_iter):
            cur = self.loss_func.derivative(x, y, w)

            if self.verbose:
                if self._cur_iter % 100 == 0:
                    print("<%s: %d iterations>"
                          % (self.__class__.__name__, self._cur_iter))

            if np.linalg.norm(cur - w) < self.eps:
                return cur
            w = cur.copy()

        return w
