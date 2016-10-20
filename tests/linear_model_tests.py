import unittest

import numpy as np

from textclassifier.core.linear_model import LinearClassifier


class LinearModelTest(unittest.TestCase):
    def test_1(self):
        """ Тестирование метода "train" класса LinearClassifier"""
        xtest = np.array([])
        ytest = np.array([])
        linear_classifier = LinearClassifier()
        with self.assertRaises(ValueError):
            linear_classifier.train(xtest, ytest)

    def test_2(self):
        """ Тестирование метода "train" класса LinearClassifier"""
        xtest = np.array([[1, 2], [3, 4]])
        ytest = np.array([1, -1, 1, -1])
        linear_classifier = LinearClassifier()
        with self.assertRaises(AttributeError):
            linear_classifier.train(xtest, ytest)

    def test_3(self):
        """ Тестирование метода "predict" класса LinearClassifier"""
        xtest = np.array([[1, 2], [3, 4]])
        ytest = np.array([1, -1])
        expected_result = np.array([-1, -1])
        linear_classifier = LinearClassifier()
        linear_classifier.train(xtest, ytest)
        true_result = linear_classifier.predict(xtest)
        for i in range(0, expected_result.shape[0]):
            self.assertAlmostEqual(expected_result[i], true_result[i], 6)

    def test_4(self):
        """ Тестирование метода "predict" класса LinearClassifier"""
        xtest = np.array([[1, 3], [3, 4], [2, -1]])
        ytest = np.array([1, -1, 1])
        expected_result = np.array([-1, -1, 1])
        linear_classifier = LinearClassifier()
        linear_classifier.train(xtest, ytest)
        true_result = linear_classifier.predict(xtest)
        for i in range(0, expected_result.shape[0]):
            self.assertAlmostEqual(expected_result[i], true_result[i], 6)

    def test_5(self):
        """ Тестирование метода "predict_probability" класса LinearClassifier"""
        xtest = np.array([[1, 2], [3, 4]])
        ytest = np.array([1, -1])
        expected_result = np.array([0.423250302588, 0.278350983928])
        linear_classifier = LinearClassifier()
        linear_classifier.train(xtest, ytest)
        true_result = linear_classifier.predict_probability(xtest)
        for i in range(0, expected_result.shape[0]):
            self.assertAlmostEqual(expected_result[i], true_result[i], 6)

    def test_6(self):
        """ Тестирование метода "predict_probability" класса LinearClassifier"""
        xtest = np.array([[1, 3], [3, 4], [2, -1]])
        ytest = np.array([1, -1, 1])
        expected_result = np.array([0.4272202419, 0.40635676, 0.6105172485956])
        linear_classifier = LinearClassifier()
        linear_classifier.train(xtest, ytest)
        true_result = linear_classifier.predict_probability(xtest)
        for i in range(0, expected_result.shape[0]):
            self.assertAlmostEqual(expected_result[i], true_result[i], 6)
