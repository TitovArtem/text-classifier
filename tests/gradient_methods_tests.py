import unittest

import numpy as np

from textclassifier.core.gradient_methods import GradientDescent


class GradientMethodTest(unittest.TestCase):
    def test_1(self):
        """Тестирование метода "transform" класса GradientDescent. """
        x_test = np.array([[0, 0], [0, 0]])
        y_test = np.array([[1, -1], [-1, 1]])
        expected_result = np.array([[0., 0.], [0., 0.]])
        gradient_descent = GradientDescent()
        true_result = gradient_descent.transform(x_test, y_test)
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(expected_result[i, j], true_result[i, j],
                                       6)

    def test_2(self):
        """Тестирование метода "transform" класса GradientDescent. """
        x_test = np.array([[3, 2], [5, 3]])
        y_test = np.array([[1, 1], [1, 1]])
        expected_result = np.array([[0.40504254, 0.40504254],
                                    [0.25597188, 0.25597188]])
        gradient_descent = GradientDescent()
        true_result = gradient_descent.transform(x_test, y_test)
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(expected_result[i, j],
                                       true_result[i, j], 6)

    def test_3(self):
        """Тестирование метода "transform" класса GradientDescent. """
        x_test = np.array([[4, 2, -2], [5, -1, 3], [1, 1, 1]])
        y_test = np.array([[1, 1, -1], [-1, -1, -1], [1, -1, 1]])
        expected_result = np.array([
            [0.00662571, -0.05068816, -0.27369039],
            [0.1904658, 0.08438458, 0.0180461],
            [-0.17102563, -0.26588115, 0.02938591]
        ])
        gradient_descent = GradientDescent()
        true_result = gradient_descent.transform(x_test, y_test)
        for i in range(0, 3):
            for j in range(0, 3):
                self.assertAlmostEqual(expected_result[i, j],
                                       true_result[i, j], 6)

    def test_4(self):
        """Тестирование метода "transform" класса GradientDescent. """
        x_test = np.array([[3, 1, 1], [-1, 2, 4], [-1, 2, 0]])
        y_test = np.array([[1, -1, -1], [1, 1, -1], [1, 1, 1]])
        expected_result = np.array([
            [0.05297498, -0.2367568, -0.14729765],
            [0.22318436, 0.13044636, -0.02462107],
            [0.20051567, 0.11687073, -0.22107901]])
        gradient_descent = GradientDescent()
        true_result = gradient_descent.transform(x_test, y_test)
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(expected_result[i, j],
                                       true_result[i, j], 6)

    def test_5(self):
        """Тестирование метода "transform" класса GradientDescent. """
        x_test = np.array([[0, 0, 0], [-2, 4, 3], [2, 3, 1]])
        y_test = np.array([[1, 1, 1], [1, 1, 1], [1, -1, 1]])
        expected_result = np.array([[0.01347941, -0.19436007, 0.01347941],
                                    [0.26164077, 0.02149885, 0.26164077],
                                    [0.14469494, 0.08169936, 0.14469494]])
        gradient_descent = GradientDescent()
        true_result = gradient_descent.transform(x_test, y_test)
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(expected_result[i, j],
                                       true_result[i, j], 6)


if __name__ == "__main__":
    unittest.main()
