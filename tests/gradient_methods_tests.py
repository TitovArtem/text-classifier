import unittest

import numpy as np
from mock import patch

from textclassifier.core.gradient_methods import GradientDescent


class GradientMethodTest(unittest.TestCase):
    def test_1(self):
        """Тестирование метода "transform" класса GradientDescent. """

        xtest = np.array([])
        ytest = np.array([])
        gradient_descent = GradientDescent()
        with self.assertRaises(IndexError):
            gradient_descent.transform(xtest, ytest)

    def test_2(self):
        """Тестирование метода "transform" класса GradientDescent. """

        x_test = np.array([[0, 0], [0, 0]])
        y_test = np.array([[1, -1], [-1, 1]])
        expected_result = np.array([[0., 0.], [0., 0.]])
        gradient_descent = GradientDescent()
        with patch('textclassifier.core.loss_functions.LogLossFunction.derivative') as der_mock:
            der_mock.return_value = np.array([[-0., -0.], [-0., -0.]])
            true_result = gradient_descent.transform(x_test, y_test)
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(expected_result[i, j], true_result[i, j],
                                       6)

    def test_3(self):
        """Тестирование метода "transform" класса GradientDescent. """

        x_test = np.array([[3, 2], [5, 3]])
        y_test = np.array([[1, 1], [1, 1]])
        expected_result = np.array([
            [0.00497063, 0.00497063],
            [0.00310685, 0.00310685],
        ])
        gradient_descent = GradientDescent()
        with patch('textclassifier.core.loss_functions.LogLossFunction.derivative') as der_mock:
            der_mock.side_effect = [
                np.array([[-1., -1.], [-0.625, -0.625]]),
                np.array([[-0.9970547, -0.9970547], [-0.62317969, -0.62317969]]),
                np.array([[-0.9941181, -0.9941181], [-0.62136477, -0.62136477]]),
                np.array([[-0.99119024, -0.99119024], [-0.61955524, -0.61955524]]),
                np.array([[-0.98827115, -0.98827115], [-0.61775113, -0.61775113]]),
            ]
        gradient_descent.max_iter = 5
        true_result = gradient_descent.transform(x_test, y_test)
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(expected_result[i, j],
                                       true_result[i, j], 6)

    def test_4(self):
        """Тестирование метода "transform" класса GradientDescent. """

        x_test = np.array([[4, 2, -2], [5, -1, 3], [1, 1, 1]])
        y_test = np.array([[1, 1, -1], [-1, -1, -1], [1, -1, 1]])
        expected_result = np.array([
            [2.46433646e-07, -5.53644527e-04, -2.21704337e-03],
            [1.11037067e-03, 5.54938451e-04, 4.93333523e-07],
            [-1.10987720e-03, -1.66493947e-03, 9.86196207e-07],
        ])
        gradient_descent = GradientDescent()
        with patch('textclassifier.core.loss_functions.LogLossFunction.derivative') as der_mock:
            der_mock.side_effect = [
                np.array([[-0., 0.11111111, 0.44444444],
                          [-0.22222222, -0.11111111, -0.],
                          [0.22222222, 0.33333333, -0.]]),
                np.array([[-2.46913564e-05, 1.10919753e-01, 4.43925926e-01],
                          [-2.22148148e-01, -1.11049383e-01, -4.93827152e-05],
                          [2.22098765e-01, 3.33160494e-01, -9.87653817e-05]]),
                np.array([[-4.93346921e-05, 1.10728650e-01, 4.43408041e-01],
                          [-2.22074104e-01, -1.10987672e-01, -9.87160429e-05],
                          [2.21975375e-01, 3.32987775e-01, -1.97385057e-04]]),
                np.array([[-7.39300648e-05, 1.10537803e-01, 4.42890790e-01],
                          [-2.22000091e-01, -1.10925980e-01, -1.48000032e-04],
                          [2.21852050e-01, 3.32815176e-01, -2.95858916e-04]]),
                np.array([[-9.84775323e-05, 1.10347210e-01, 4.42374173e-01],
                          [-2.21926107e-01, -1.10864305e-01, -1.97234732e-04],
                          [2.21728790e-01, 3.32642697e-01, -3.94186852e-04]])
            ]
            gradient_descent.max_iter = 5
            true_result = gradient_descent.transform(x_test, y_test)
        for i in range(0, 3):
            for j in range(0, 3):
                self.assertAlmostEqual(expected_result[i, j],
                                       true_result[i, j], 6)

    def test_5(self):
        """Тестирование метода "transform" класса GradientDescent. """

        x_test = np.array([[3, 1, 1], [-1, 2, 4], [-1, 2, 0]])
        y_test = np.array([[1, -1, -1], [1, 1, -1], [1, 1, 1]])
        expected_result = np.array([
            [0.00027776, -0.00138795, -0.00083292],
            [0.00138752, 0.00083242, -0.00027699],
            [0.0013869, 0.00083205, -0.00138749],
        ])
        gradient_descent = GradientDescent()
        with patch('textclassifier.core.loss_functions.LogLossFunction.derivative') as der_mock:
            der_mock.side_effect = [
                np.array([
                    [-0.05555556, 0.27777778, 0.16666667],
                    [-0.27777778, -0.16666667, 0.05555556],
                    [-0.27777778, -0.16666667, 0.27777778],
                ]),
                np.array([
                    [-0.05555401, 0.27768364, 0.166625],
                    [-0.27764043, -0.16657562, 0.05547685],
                    [-0.2775787, -0.16653858, 0.27763735],
                ]),
                np.array([
                    [-0.05555246, 0.27758954, 0.16658334],
                    [-0.27750317, -0.16648463, 0.0553982],
                    [-0.27737976, -0.16641058, 0.277497]
                ]),
                np.array([
                    [-0.0555509, 0.27749548, 0.16654169],
                    [-0.27736599, -0.16639369, 0.05531961],
                    [-0.27718094, -0.16628267, 0.27735674]
                ]),
                np.array([
                    [-0.05554933, 0.27740144, 0.16650004],
                    [-0.2772289, -0.16630281, 0.05524106],
                    [-0.27698225, -0.16615484, 0.27721656]
                ]),
            ]
            gradient_descent.max_iter = 5
            true_result = gradient_descent.transform(x_test, y_test)
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(expected_result[i, j],
                                       true_result[i, j], 6)

    def test_6(self):
        """Тестирование метода "transform" класса GradientDescent. """

        x_test = np.array([[0, 0, 0], [-2, 4, 3], [2, 3, 1]])
        y_test = np.array([[1, 1, 1], [1, 1, 1], [1, -1, 1]])
        expected_result = np.array([
            [4.62406435e-07, -1.11046325e-03, 4.62406435e-07],
            [1.94082139e-03, 2.76806595e-04, 1.94082139e-03],
            [1.10887565e-03, 5.54769215e-04, 1.10887565e-03],
        ])
        gradient_descent = GradientDescent()
        with patch('textclassifier.core.loss_functions.LogLossFunction.derivative') as der_mock:
            der_mock.side_effect = [
                np.array([
                    [-0., 0.22222222, -0.],
                    [-0.38888889, -0.05555556, -0.38888889],
                    [-0.22222222, -0.11111111, -0.22222222],
                ]),
                np.array([
                    [-4.62962579e-05, 2.22157407e-01, -4.62962579e-05],
                    [-3.88526235e-01, -5.54583333e-02, -3.88526235e-01],
                    [-2.21998457e-01, -1.11032407e-01, -2.21998457e-01],
                ]),
                np.array([
                    [-9.25369875e-05, 2.22092621e-01, -9.25369875e-05],
                    [-3.88163929e-01, -5.53612151e-02, -3.88163929e-01],
                    [-2.21774910e-01, -1.10953773e-01, -2.21774910e-01],
                ]),
                np.array([
                    [-1.38722016e-04, 2.22027864e-01, -1.38722016e-04],
                    [-3.87801972e-01, -5.52642008e-02, -3.87801972e-01],
                    [-2.21551583e-01, -1.10875209e-01, -2.21551583e-01],
                ]),
                np.array([
                    [-1.84851173e-04, 2.21963134e-01, -1.84851173e-04],
                    [-3.87440364e-01, -5.51672903e-02, -3.87440364e-01],
                    [-2.21328475e-01, -1.10796714e-01, -2.21328475e-01],
                ]),
            ]
            gradient_descent.max_iter = 5
            true_result = gradient_descent.transform(x_test, y_test)

        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(expected_result[i, j],
                                       true_result[i, j], 6)


if __name__ == "__main__":
    unittest.main()
