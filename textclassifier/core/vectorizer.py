import collections
from functools import reduce

import numpy as np

from textclassifier.core.base import BaseEstimator
from textclassifier.core.preprocessing.text import TextToWordsConverter


class TfidfVectorizer(BaseEstimator):
    """ Convert a texts to TF-IDF features. """

    def __init__(self, to_words_converter=TextToWordsConverter(),
                 max_features=None):
        self._idf = {}
        self._converter = to_words_converter
        self.max_features = max_features

    @property
    def text_to_words_converter(self):
        return self._converter

    @text_to_words_converter.setter
    def text_to_words_converter(self, converter):
        if converter is None:
            raise ValueError("The given converter is none.")
        self._converter = converter

    def _get_texts_words(self, text):
        return self._converter.convert(text)

    def _count_idf(self, texts):
        id = 0
        for text in texts:
            for word in text:
                # if _idf already has this word
                if self._idf.get(word):
                    continue

                # calculate the number of documents which has this word
                count = reduce(lambda x, y: x + 1 if word in y else x, texts, 0)
                self._idf[word] = (id, np.log10(len(texts) / count))
                id += 1

        if self.max_features:
            self._filter_features_by_max_count()

    def _filter_features_by_max_count(self):
        sorted_idf_items = sorted(self._idf.items(),
                                  key=lambda x: x[1][1], reverse=True)
        sorted_idf_items = sorted_idf_items[:self.max_features]
        # reordering idf features
        self._idf = dict((f[0], (i, f[1][1]))  # (key, (index, idf-value))
                         for i, f in enumerate(sorted_idf_items))

    @staticmethod
    def _count_tf(text_words):
        count_words = collections.Counter(text_words)
        length = float(len(count_words))
        return {key: count_words[key] / length for key in count_words}

    def _transform(self, x, texts):
        """ Count TF-IDF for each text in the given list of texts. """
        if not self._idf:
            raise ValueError("Model is not trained.")

        res = np.zeros((len(x), len(self._idf)))

        if not texts:
            texts = [self._get_texts_words(text) for text in x]

        for i, text in enumerate(texts):
            tf = TfidfVectorizer._count_tf(text)
            for k, v in tf.items():
                if not self._idf.get(k):
                    continue
                idf = self._idf[k]
                res[i, idf[0]] = idf[1] * v

        return res

    def _train(self, x):
        texts = [self._get_texts_words(text) for text in x]
        self._count_idf(texts)
        return texts

    def train(self, x):
        self._train(x)

    def transform(self, x):
        return self._transform(x=x, texts=None)

    def train_transform(self, x):
        texts = self._train(x)
        return self._transform(x=x, texts=texts)
