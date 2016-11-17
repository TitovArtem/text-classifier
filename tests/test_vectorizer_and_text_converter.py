import unittest

import numpy as np

from textclassifier.core.preprocessing.text import TextToWordsConverter
from textclassifier.core.vectorizer import TfidfVectorizer


class TfidfVectorizerTextToWordsConverterTest(unittest.TestCase):
    def dictionary_equality(self, dic_1, dic_2, is_tuple=True):
        self.assertEqual(len(dic_1), len(dic_2))
        for key, value in dic_1.items():
            self.assertTrue(key in dic_2)
            if is_tuple:
                self.assertEqual(dic_2[key][0], value[0])
                self.assertAlmostEqual(dic_2[key][1], value[1], places=12)

    def numpy_array_equality(self, arra_1, array_2):
        self.assertEqual(len(arra_1), len(array_2))
        for row in range(len(arra_1)):
            self.assertEqual(len(arra_1[row]), len(array_2[row]))

        for i in range(len(arra_1)):
            for j in range(len(arra_1[i])):
                self.assertAlmostEqual(arra_1[i][j], array_2[i][j], places=7)

    def test_1(self):
        """ Тест для метода _transform класса TfidfVectorizer. """

        test_list = [
            "the british museum has one of the largest libraries "
            "in the", "world it has a copy of every book that is "
        ]
        idf_list = [[
            'the', 'british', 'museum', 'has', 'one', 'of', 'the', 'largest',
            'libraries', 'in', 'the', 'world', 'it', 'has', 'a', 'copy', 'of',
            'every', 'book', 'that', 'is'], ['chefirovka', 'is', 'a', 'large',
                                             'village', 'not', 'far', 'from',
                                             'tula', 'the', 'people', 'who',
                                             'live', 'in', 'chefirovka', 'grow',
                                             'vegetables', 'and', 'various',
                                             'kinds', 'of']
        ]
        ex_res = np.zeros((2, 32))
        ex_res[0][1] = ex_res[0][2] = ex_res[0][3] = ex_res[0][4] = 0.04300429
        ex_res[0][6] = ex_res[0][7] = 0.04300429
        ex_res[1][3] = ex_res[1][9] = ex_res[1][10] = ex_res[1][12] = 0.03344778
        ex_res[1][13] = ex_res[1][14] = ex_res[1][15] = 0.03344778
        tf_idf = TfidfVectorizer(to_words_converter=TextToWordsConverter())
        tf_idf._count_idf(idf_list)
        true_result = tf_idf._transform(test_list, None)
        self.numpy_array_equality(true_result, ex_res)

    def test_2(self):
        """ Тест для метода _transform класса TfidfVectorizer. """

        test_list = [
            "the british museum has one of the largest libraries "
            "in the", "world it has a copy of every book that is "
        ]
        tf_idf = TfidfVectorizer(to_words_converter=TextToWordsConverter())
        with self.assertRaises(ValueError):
            tf_idf._transform(test_list, None)

    def test_3(self):
        """ Тест для метода _transform класса TfidfVectorizer. """

        test_list = []
        idf_list = [[
            'the', 'british', 'museum', 'has', 'one', 'of', 'the', 'largest',
            'libraries', 'in'], ['chefirovka', 'is', 'a', 'large',
                                 'village', 'not', 'far', 'from']
        ]
        ex_res = []
        tf_idf = TfidfVectorizer(to_words_converter=TextToWordsConverter())
        tf_idf._count_idf(idf_list)
        true_result = tf_idf._transform(test_list, None)
        self.numpy_array_equality(true_result, ex_res)

    def test_4(self):
        """ Тест для метода _transform класса TfidfVectorizer. """

        test_list = [
            "the british museum has one of the largest libraries "
            "in the", "world it has a copy of every book that is "
        ]
        idf_list = [[
            'british', 'museum', 'has', 'one', 'of', 'the', 'largest',
            'libraries', 'in'], ['chefirovka', 'is', 'a', 'large',
                                 'village', 'not', 'far', 'from']
        ]

        ex_res = np.zeros((2, 17))
        ex_res[0][0] = ex_res[0][1] = ex_res[0][2] = ex_res[0][3] = 0.04300429
        ex_res[0][4] = ex_res[0][6] = ex_res[0][7] = 0.04300429
        ex_res[1][2] = ex_res[1][4] = ex_res[1][10] = 0.03344778
        tf_idf = TfidfVectorizer(to_words_converter=TextToWordsConverter())
        tf_idf._count_idf(idf_list)
        true_result = tf_idf._transform(test_list, None)
        self.numpy_array_equality(true_result, ex_res)

    def test_5(self):
        """ Тест для метода _transform класса TfidfVectorizer. """

        true_result = [
            [0.08804563, 0.08804563, 0., 0., 0., 0., 0.],
            [0., 0.05869709, 0.15904042, 0.05869709, 0., 0., 0.],
            [0.03521825, 0., 0., 0.03521825, 0.09542425, 0.09542425, 0.09542425]
        ]

        texts = [
            "Hello world.",
            "World is so beautiful",
            "Hello Marry, you are beautiful"
        ]

        vectorizer = TfidfVectorizer(to_words_converter=TextToWordsConverter())
        res = vectorizer.train_transform(texts)
        self.numpy_array_equality(true_result, res)

if __name__ == '__main__':
    unittest.main()
