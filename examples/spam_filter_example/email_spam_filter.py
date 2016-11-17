import email

import numpy as np

from examples.spam_filter_example.data_extraction import extract_data, \
    extract_target
from textclassifier.core.gradient_methods import GradientDescent
from textclassifier.spam_filter import SpamFilter


def main():
    try:
        print("> Extracting train data")
        x, indexes = extract_data()
        y = np.array(extract_target(indexes))
    except IOError:
        print("> Can't read data from the train dataset.")
        exit(1)
    except IndexError:
        print("> Invalid size of the train dataset")
        exit(1)

    print("> Training classifier")
    filter = SpamFilter()
    filter.classifier.optimizer = GradientDescent(eps=1e-4, step=100)
    filter.train(x, y)
    print("> Classifier is trained")

    while True:
        try:
            email_filename = input("> Input filename with email data: ")
            with open(email_filename) as file:
                msg = email.message_from_file(file).get_payload()
                if isinstance(msg, list):
                    msg = msg[0].get_payload()
                if isinstance(msg, list):
                    print("> Invalid message format.")
                    continue
                if not msg:
                    print("> Message is empty.")
                    continue
                is_spam, prob = filter.is_spam(msg, get_prob=True)
                if is_spam:
                    print("> Message is spam. Assurance: %f" % prob)
                else:
                    print("> Message is not spam. Assurance: %f" % (1.0 - prob))
        except FileNotFoundError:
            print("> File '%s' doesn't exist." % email_filename)
        except UnicodeDecodeError:
            print("> File '%s' has unicode errors." % email_filename)
        except IOError:
            print("> Error of reading file '%s'." % email_filename)


if __name__ == '__main__':
    main()
