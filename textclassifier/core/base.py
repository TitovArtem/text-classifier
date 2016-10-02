import copy
from abc import abstractmethod, ABCMeta


class BaseEstimator(object):
    """ The base class for all estimators. """

    def __init__(self, **params):
        self._params = params if params else {}

    @property
    def params(self):
        """ Return the dictionary of parameters for this estimator. """
        return self._params

    @params.setter
    def params(self, **params):
        """ Set the parameters for this estimator. """
        if not params:
            return

        self._check_params(**params)
        for key, val in params:
            self._params[key] = copy.copy(val)

    def _check_params(self, **params):
        for p in params.keys():
            if p not in self._params.keys():
                raise ValueError("Invalid parameter %s for estimator %s." %
                                 (p, self))


class UnsupervisedModel(BaseEstimator, metaclass=ABCMeta):
    """ The base class for models without learning. """

    @abstractmethod
    def predict(self, x):
        """ Predict results for x. """


class SupervisedModel(UnsupervisedModel, metaclass=ABCMeta):
    """ The base class for models with supervised learning. """

    @abstractmethod
    def train(self, x, y):
        """ Train model. """
