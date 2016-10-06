import copy
from abc import abstractmethod, ABCMeta


class BaseEstimator(object):
    """ The base class for all estimators. """

    def get_params(self):
        """ Return the dictionary of public parameters for this estimator. """
        return dict((k, v) for k, v in self.__dict__.items()
                    if not k.startswith("_"))

    def set_params(self, **params):
        """ Set the public parameters for this estimator. """
        if not params:
            return

        self._check_params(**params)
        for key, val in params.items():
            self.__dict__[key] = copy.deepcopy(val)

    def _check_params(self, **params):
        for p in params.keys():
            if p not in self.get_params():
                raise ValueError("Invalid parameter '%s' for estimator %s." %
                                 (p, self))

    def __repr__(self):
        return "%s: %s" % (self.__class__.__name__, self.get_params())


class UnsupervisedModel(BaseEstimator, metaclass=ABCMeta):
    """ The base class for models based on unsupervised learning. """

    @abstractmethod
    def predict(self, x):
        """
        Predict results for x.
        :param x: numpy array, test data
        :return: numpy array, predicted answers
        """


class SupervisedModel(UnsupervisedModel, metaclass=ABCMeta):
    """ The base class for models based on supervised learning. """

    @abstractmethod
    def train(self, x, y):
        """
        Train model.
        :param x: numpy array, train data
        :param y: numpy array, target data
        """
