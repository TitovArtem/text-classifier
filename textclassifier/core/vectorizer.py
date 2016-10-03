import math
import collections
import numpy as np

from textclassifier.core.preprocessing.text import SimpleTextSplitter, \
    TextFilter


class TfidfVectorizer(object):
    """ Convert a texts to TF-IDF features. """

    def __init__(self, splitter=SimpleTextSplitter(),
                 preprocessor=TextFilter()):
        self._splitter = splitter
        self._preprocessor = preprocessor
        self._idf_mas = {}

    @property
    def splitter(self):
        return self._splitter

    @splitter.setter
    def splitter(self, value):
        if not value:
            raise ValueError("The given splitter can't be none.")
        self._splitter = value

    @property
    def preprocessor(self):
        return self._preprocessor

    @preprocessor.setter
    def preprocessor(self, value):
        if not value:
            raise ValueError("The given preprocessor can't be none.")
        self._preprocessor = value

    def _count_tf(self, text):
        count_words = collections.Counter(text)
        length = float(len(count_words))
        return {key: count_words[key] / length for key in count_words}

    def _count_idf(self, texts):
        for text in texts:
            for word in text:
                if word not in self._idf_mas:
                    b = sum([1.0 for item in texts if word in item])
                    self._idf_mas[word] = math.log10(len(texts) / b)

    def _get_texts_words(self, texts):
        texts_words = []
        for text in texts:
            words = self._splitter.split(text)
            texts_words.append(self._preprocessor.transform(words))
        return texts_words

    def transform(self, texts):
        texts_words = self._get_texts_words(texts)
        self._count_idf(texts_words)
        tfidf_dic = {}
        for text in texts_words:
            temp = self._count_tf(text)
            tfidf_dic.update({word: temp[word] *
                self._idf_mas[word] for word in temp})

        temp_list = [0] * len(texts_words)
        for i, text in enumerate(texts_words):
            for word in text:
                temp_list[i] += tfidf_dic[word]
        return np.array(temp_list)
