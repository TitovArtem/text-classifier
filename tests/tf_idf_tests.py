from math import fabs
import unittest
import numpy as np
from textclassifier.core.vectorizer import TfidfVectorizer


class TfidfVectorizerTest(unittest.TestCase):
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
        """ Тест для метода _count_idf класса TfidfVectorizer. """
        test_list = [
            ['expect', 'think', 'london', 'very', 'pattern', 'little',
             'pattern', 'because', 'from', 'air',
             'see', 'pattern', 'kind', 'picture', 'which', 'streets', 'look',
             'think', 'tiny', 'compared',
             'mass', 'buildings'],
            ['right', 'london', 'no', 'think', 'pattern', 'because', 'past',
             'spread', 'every',
             'direction', 'one', 'century'],
            ['built', 'fields', 'west', 'picture', 'direction', 'another',
             'south', 'london', 'think', 'direction', 'yet', 'another', 'north',
             'east']
        ]
        tf_idf = TfidfVectorizer()
        tf_idf._count_idf(test_list)
        expect_result = tf_idf._idf
        true_result = {'expect': (0, 0.4771212547196623), 'think': (1, 0.0),
                       'london': (2, 0.0),
                       'very': (3, 0.47712125471966244),
                       'pattern': (4, 0.17609125905568124),
                       'little': (5, 0.4771212547196623),
                       'because': (6, 0.17609125905568124),
                       'from': (7, 0.4771212547196623),
                       'air': (8, 0.4771212547196623),
                       'see': (9, 0.4771212547196623),
                       'kind': (10, 0.4771212547196623),
                       'picture': (11, 0.17609125905568124),
                       'which': (12, 0.4771212547196623),
                       'streets': (13, 0.4771212547196623),
                       'look': (14, 0.4771212547196623),
                       'tiny': (15, 0.4771212547196623),
                       'compared': (16, 0.4771212547196623),
                       'mass': (17, 0.4771212547196623),
                       'buildings': (18, 0.4771212547196623),
                       'right': (19, 0.4771212547196623),
                       'no': (20, 0.4771212547196623),
                       'past': (21, 0.4771212547196623),
                       'spread': (22, 0.4771212547196623),
                       'every': (23, 0.4771212547196623),
                       'direction': (24, 0.17609125905568124),
                       'one': (25, 0.4771212547196623),
                       'century': (26, 0.4771212547196623),
                       'built': (27, 0.4771212547196623),
                       'fields': (28, 0.4771212547196623),
                       'west': (29, 0.4771212547196623),
                       'another': (30, 0.4771212547196623),
                       'south': (31, 0.4771212547196623),
                       'yet': (32, 0.4771212547196623),
                       'north': (33, 0.4771212547196623),
                       'east': (34, 0.4771212547196623)}
        self.dictionary_equality(expect_result, true_result)

    def test_2(self):
        """ Тест для метода _count_idf класса TfidfVectorizer. """
        test_list = [[], [], [], []]
        tf_idf = TfidfVectorizer()
        tf_idf._count_idf(test_list)
        expect_result = tf_idf._idf
        true_result = {}
        self.dictionary_equality(expect_result, true_result)

    def test_3(self):
        """ Тест для метода _count_idf класса TfidfVectorizer. """
        test_list = [
            ['expect', 'think', 'london', 'very', 'pattern', 'little',
             'pattern', 'because', 'from', 'london', 'very',
             'pattern']]
        tf_idf = TfidfVectorizer()
        tf_idf._count_idf(test_list)
        expect_result = tf_idf._idf
        true_result = {'expect': (0, 0.0), 'think': (1, 0.0),
                       'london': (2, 0.0), 'very': (3, 0.0),
                       'pattern': (4, 0.0),
                       'little': (5, 0.0),
                       'because': (6, 0.0), 'from': (7, 0.0)}
        self.dictionary_equality(expect_result, true_result)

    def test_4(self):
        """ Тест для метода _count_tf класса TfidfVectorizer. """
        test_list = ['but', 'how', 'is', 'dull', 'if', 'to', 'every', 'street',
                     'straight', 'if', 'every', 'open',
                     'space', 'were', 'square',
                     'knew', 'what', 'you', 'to', 'were', 'going', 'to', 'find',
                     'is', 'wherever', 'you', 'went',
                     'fortunately', 'london',
                     'is', 'not', 'like', 'that', 'it', 'is', 'a', 'mixture',
                     'of', 'fine', 'streets', 'side', 'is',
                     'by', 'side', 'with', 'narrow',
                     'courts', 'and', 'of', 'really', 'facing', 'those',
                     'built', 'so', 'to', 'say']
        tf_idf = TfidfVectorizer()
        expect_result = tf_idf._count_tf(test_list)
        one = 0.023255813953488372
        two = 0.046511627906976744
        four = 0.09302325581395349
        five = 0.11627906976744186
        true_result = {'but': one, 'is': five, 'how': one, 'dull': one,
                       'if': two, 'to': four, 'every': two,
                       'street': one, 'straight': one, 'open': one,
                       'space': one, 'were': two, 'square': one, 'knew': one,
                       'what': one, 'you': two, 'going': one,
                       'find': one, 'wherever': one, 'went': one,
                       'fortunately': one, 'london': one, 'not': one,
                       'like': one, 'that': one, 'it': one, 'a': one,
                       'mixture': one, 'fine': one, 'streets': one,
                       'side': two, 'by': one, 'with': one, 'narrow': one,
                       'courts': one, 'and': one, 'of': two,
                       'really': one, 'facing': one, 'those': one, 'built': one,
                       'so': one, 'say': one}
        self.assertDictEqual(expect_result, true_result)

    def test_5(self):
        """ Тест для метода _count_tf класса TfidfVectorizer. """
        test_list = []
        tf_idf = TfidfVectorizer()
        expect_result = tf_idf._count_tf(test_list)
        true_result = {}
        self.assertDictEqual(expect_result, true_result)

    def test_6(self):
        """ Тест для метода _count_tf класса TfidfVectorizer. """
        test_list = ['mixture', 'mixture', 'mixture', 'mixture', 'mixture',
                     'mixture', 'mixture', 'mixture', 'mixture',
                     'mixture', 'mixture', 'mixture', 'mixture', 'mixture']
        tf_idf = TfidfVectorizer()
        expect_result = tf_idf._count_tf(test_list)
        true_result = {'mixture': 14.0}
        self.assertDictEqual(expect_result, true_result)

    def test_7(self):
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

        tf_idf = TfidfVectorizer()
        tf_idf._count_idf(idf_list)
        true_result = tf_idf._transform(test_list, None)
        self.numpy_array_equality(true_result, ex_res)

    def test_8(self):
        """ Тест для метода _transform класса TfidfVectorizer. """
        test_list = [
            "the british museum has one of the largest libraries "
            "in the", "world it has a copy of every book that is "
        ]
        tf_idf = TfidfVectorizer()
        with self.assertRaises(ValueError):
            tf_idf._transform(test_list, None)

    def test_9(self):
        """ Тест для метода _transform класса TfidfVectorizer. """
        test_list = []
        idf_list = [[
            'the', 'british', 'museum', 'has', 'one', 'of', 'the', 'largest',
            'libraries', 'in'], ['chefirovka', 'is', 'a', 'large',
                                 'village', 'not', 'far', 'from']
        ]
        ex_res = []
        tf_idf = TfidfVectorizer()
        tf_idf._count_idf(idf_list)
        true_result = tf_idf._transform(test_list, None)
        self.numpy_array_equality(true_result, ex_res)

if __name__ == '__main__':
    unittest.main()
