import re
from abc import ABCMeta, abstractmethod


ENGLISH_PREPOSITIONS = [
    "on", "in", "at", "since", "for", "ago", "before", "to", "past", "till",
    "by", "a", "an", "the", "but", "as", "over", "so", "out", "below" "there",
]

ENGLISH_PRONOUNS = [
    "i", "you", "she", "he", "it", "we", "they" "me", "his", "her", "him", "us",
    "them", "their", "our", "your",
]


ENGLISH_AUXILIARY_VERBS = [
    "be", "am", "are", "is", "was", "were", "being", "been", "can", "could",
    "dare", "do", "does", "did", "have", "has", "having", "may", "might",
    "must", "need", "shall", "would", "will", "should"
]


ENGLISH_STOP_WORDS = ENGLISH_PREPOSITIONS + ENGLISH_PRONOUNS + \
                     ENGLISH_AUXILIARY_VERBS


class AbstractTextSplitter(metaclass=ABCMeta):
    """ Abstract class for all text splitters. """

    @abstractmethod
    def split(self, text):
        """ Split text by words. """


class SimpleTextSplitter(AbstractTextSplitter):

    def split(self, text):
        return re.findall(r'\w+', text)


class AbstractTextPreprocessor(metaclass=ABCMeta):
    """ Abstract class for all text transformers. """

    @abstractmethod
    def transform(self, words):
        """ Transform the given list of words. """


class TextFilter(AbstractTextPreprocessor):
    """ The filter for deleting redundant words from text. """

    def __init__(self, stop_words=ENGLISH_PREPOSITIONS):
        self._stop_words = stop_words[:]

    @property
    def stop_words(self):
        return self._stop_words

    @stop_words.setter
    def stop_words(self, words):
        if words:
            self._stop_words = words[:]

    def transform(self, words):
        return [w.lower() for w in words if w not in self._stop_words]


class AbstractTextToWordsConverter(metaclass=ABCMeta):
    """ Abstract class for converting text to list of words. """

    @abstractmethod
    def convert(self, text):
        """ Convert text to list of words. """


class TextToWordsConverter(AbstractTextToWordsConverter):

    def __init__(self, splitter=SimpleTextSplitter(), transformer=TextFilter()):
        self._splitter = splitter
        self._transformer = transformer

    @property
    def splitter(self):
        return self._splitter

    @splitter.setter
    def splitter(self, other):
        if other:
            self._splitter = other

    @property
    def transformer(self):
        return self._transformer

    @transformer.setter
    def transformer(self, other):
        if other:
            self._transformer = other

    def convert(self, text):
        words = self._splitter.split(text)
        return self._transformer.transform(words)
