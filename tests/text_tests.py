import unittest
from textclassifier.core.preprocessing.text import SimpleTextSplitter, \
    TextFilter, ENGLISH_STOP_WORDS


class TfidfVectorizerTest(unittest.TestCase):
    def test_1(self):
        """ Тестирование метода "split" класса SimpleTextSplitter. """
        test_list = "If you... flew() over: London you would see the city" \
                    " spread out below you. There would be the Monument," \
                    " the tall! 99 buildings in, the, City, packed so???" \
                    " closely together."
        expected_result = ['If', 'you', 'flew', 'over', 'London', 'you',
                           'would', 'see', 'the', 'city', 'spread', 'out',
                           'below', 'you', 'There', 'would', 'be', 'the',
                           'Monument', 'the', 'tall', '99', 'buildings',
                           'in', 'the', 'City', 'packed',
                           'so', 'closely', 'together']
        splitter = SimpleTextSplitter()
        true_result = splitter.split(test_list)
        self.assertListEqual(expected_result, true_result)

    def test_2(self):
        """ Тестирование метода "split" класса SimpleTextSplitter. """
        test_list = "!@#$%^&*().,"
        expected_result = []
        splitter = SimpleTextSplitter()
        true_result = splitter.split(test_list)
        self.assertListEqual(expected_result, true_result)

    def test_3(self):
        """ Тестирование метода "split" класса SimpleTextSplitter. """
        test_list = ""
        expected_result = []
        splitter = SimpleTextSplitter()
        true_result = splitter.split(test_list)
        self.assertListEqual(expected_result, true_result)

    def test_4(self):
        """ Тестирование метода "transform" класса TextFilter. """
        test_list = ['If', 'you', 'flew', 'over', 'London', 'you', 'would',
                     'see', 'the', 'city', 'spread', 'out', 'below',
                     'you', 'There', 'would', 'be', 'the', 'Monument', 'the',
                     'tall', '99', 'buildings', 'in', 'the', 'City', 'packed',
                     'so', 'closely', 'together']
        expected_result = ['if', 'flew', 'london', 'see', 'city', 'spread',
                           'below', 'there', 'monument', 'tall', '99',
                           'buildings',
                           'city', 'packed', 'closely', 'together']
        text_filter = TextFilter()
        text_filter.stop_words = ENGLISH_STOP_WORDS
        true_result = text_filter.transform(test_list)
        self.assertListEqual(expected_result, true_result)

    def test_5(self):
        """ Тестирование метода "transform" класса TextFilter. """
        test_list = ['If', 'you', 'be', 'over', 'us', 'you', 'would', 'he',
                     'the', 'He', 'She', 'Was', 'their', 'a',
                     'A']
        expected_result = ['if', 'he', 'she', 'was', 'a']
        text_filter = TextFilter()
        text_filter.stop_words = ENGLISH_STOP_WORDS
        true_result = text_filter.transform(test_list)
        self.assertListEqual(expected_result, true_result)

    def test_6(self):
        """ Тестирование метода "transform" класса TextFilter. """
        test_list = []
        expected_result = []
        text_filter = TextFilter()
        text_filter.stop_words = ENGLISH_STOP_WORDS
        true_result = text_filter.transform(test_list)
        self.assertListEqual(expected_result, true_result)


if __name__ == '__main__':
    unittest.main()
