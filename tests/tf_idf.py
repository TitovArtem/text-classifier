from math import fabs
import unittest
from textclassifier.core.vectorizer import TfidfVectorizer


class TfidfVectorizerTest(unittest.TestCase):
    def dictionary_equality(self, dic_1, dic_2, is_tuple=True):
        self.assertEqual(len(dic_1), len(dic_2))
        for key, value in dic_1.items():
            self.assertTrue(key in dic_2)
            if is_tuple:
                self.assertEqual(dic_2[key][0], value[0])
                self.assertAlmostEqual(dic_2[key][1], value[1], places=12)

    def test_1(self):
        """ Тесты для метода _count_idf класса TfidfVectorizer. """
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
        """ Тесты для метода _count_idf класса TfidfVectorizer. """
        test_list = [[], [], [], []]
        tf_idf = TfidfVectorizer()
        tf_idf._count_idf(test_list)
        expect_result = tf_idf._idf
        true_result = {}
        self.dictionary_equality(expect_result, true_result)

    def test_3(self):
        """ Тесты для метода _count_idf класса TfidfVectorizer. """
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
        """ Тесты для метода _count_tf класса TfidfVectorizer. """
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
        """ Тесты для метода _count_tf класса TfidfVectorizer. """
        test_list = []
        tf_idf = TfidfVectorizer()
        expect_result = tf_idf._count_tf(test_list)
        true_result = {}
        self.assertDictEqual(expect_result, true_result)

    def test_6(self):
        """ Тесты для метода _count_tf класса TfidfVectorizer. """
        test_list = ['mixture', 'mixture', 'mixture', 'mixture', 'mixture',
                     'mixture', 'mixture', 'mixture', 'mixture',
                     'mixture', 'mixture', 'mixture', 'mixture', 'mixture']
        tf_idf = TfidfVectorizer()
        expect_result = tf_idf._count_tf(test_list)
        true_result = {'mixture': 14.0}
        self.assertDictEqual(expect_result, true_result)


if __name__ == '__main__':
    unittest.main()
