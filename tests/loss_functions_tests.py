import unittest
from textclassifier.core.loss_functions import LogLossFunction
import numpy as np

class TfidfVectorizerTest(unittest.TestCase):
    def test_1(self):
        """ Тестирование метода "value" класса LogLossFunction. """
        expected_result = 1.0
        loss_function = LogLossFunction()
        true_result = loss_function.value(0, 0, 0)
        self.assertEqual(expected_result, true_result)

    def test_2(self):
        """ Тестирование метода "value" класса LogLossFunction. """
        expected_result = 0.0035716586710060967
        loss_function = LogLossFunction()
        true_result = loss_function.value(3, 2, 1)
        self.assertEqual(expected_result, true_result)

    def test_3(self):
        """ Тестирование метода "value" класса LogLossFunction. """
        expected_result = 0.0
        loss_function = LogLossFunction()
        true_result = loss_function.value(5, 8, 3)
        self.assertEqual(expected_result, true_result)

    def test_4(self):
        """ Тестирование метода "derivate" класса LogLossFunction. """
        xtest = np.array ([[1, 2], [4, 2]])
        ytest = np.array ([[-1, 1], [-1, 1]])
        wtest = np.array ([3, 4])
        expect_result = np.array ([[1.24997912e+00, -2.57644212e-09],[9.99983299e-01, -2.06115369e-09]])
        loss_function = LogLossFunction()
        true_result = loss_function.derivative(xtest, ytest, wtest)
        for i in range(0, 2):
            for j in range(0, 2):
                self.assertAlmostEqual(true_result[i, j], expect_result[i, j], 6)

    def test_5(self):
        """ Тестирование метода "derivate" класса LogLossFunction. """
        xtest = np.array ([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        ytest = np.array ([[-1, 1, 1], [-1, 1, -1], [1, 1, 1]])
        wtest = np.array ([3, 1, 2])
        expect_result = np.array ([[0, 0, 0],[0, 0, 0], [0, 0, 0]])
        loss_function = LogLossFunction()
        true_result = loss_function.derivative(xtest, ytest, wtest)
        for i in range(0, 3):
            for j in range(0, 3):
                self.assertAlmostEqual(true_result[i, j], expect_result[i, j], 6)

    def test_6(self):
        """ Тестирование метода "derivate" класса LogLossFunction. """
        xtest = np.array ([[0, 1, 2], [0, 5, 3], [6, 4, 8]])
        ytest = np.array ([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        wtest = np.array ([4, 4, 4])
        expect_result = np.array ([[-4.09611640e-06, -8.43769499e-15, -0.00000000e+00],
                                   [-6.82686067e-06, -1.40628250e-14, -0.00000000e+00],
                                   [-8.87491887e-06, -1.82816725e-14, -0.00000000e+00]])
        loss_function = LogLossFunction()
        true_result = loss_function.derivative(xtest, ytest, wtest)
        for i in range(0, 3):
            for j in range(0, 3):
                self.assertAlmostEqual(true_result[i, j], expect_result[i, j], 6)