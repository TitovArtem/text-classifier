import numpy as np

from textclassifier.core.base import ClassifierMixin, \
    SupervisedModel, AttributeTypeError
from textclassifier.core.linear_model import LinearClassifier
from textclassifier.core.vectorizer import TfidfVectorizer


class SpamFilter(SupervisedModel):
    
    def __init__(self, classifier=LinearClassifier(), 
                 vectorizer=TfidfVectorizer(), assurance=0.5):
        self.classifier = classifier
        self.vectorizer = vectorizer
        self.assurance = assurance
        self._validate_params()
        
    def _validate_params(self):
        if not isinstance(self.classifier, (ClassifierMixin, SupervisedModel)):
            raise ValueError("Attribute 'classifier' is invalid.")

        if self.vectorizer is None:
            raise ValueError("Attribute 'vectorizer' is None.")

        if not isinstance(self.assurance, (float, int)):
            raise AttributeTypeError("assurance", float.__name__)

        if self.assurance < 0.0 or self.assurance > 1.0:
            raise ValueError("Attribute assurance must be in range [0.0, 1.0].")

    def train(self, x, y):
        self._validate_params()
        train_x = self.vectorizer.train_transform(x)
        y = np.copy(y)
        y[y < 0.0] = -1.0
        y[y >= 0.0] = 1.0
        self.classifier.train(train_x, y)

    def predict(self, x):
        self._validate_params()
        x = self.vectorizer.transform(x)
        return self.classifier.predict(x)

    def is_spam(self, text, get_prob=False):
        x = self.vectorizer.transform([text])
        prob = self.classifier.predict_probability(x)

        if get_prob:
            return prob >= self.assurance, prob
        return prob >= self.assurance
