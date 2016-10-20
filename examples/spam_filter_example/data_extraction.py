import email.parser

TRAIN_FILE = "TRAIN"

TRAIN_PATH = "dataset/train/"
TARGET_PATH = "dataset/target.txt"


def construct_filename(file_id, type_pattern=TRAIN_FILE):
    str_id = str(file_id)
    file_number = (5 - len(str_id)) * "0" + str_id
    return type_pattern + "_" + file_number + ".eml"


def extract_data(files_num=150, files_path=TRAIN_PATH, type_pattern=TRAIN_FILE):
    res = []
    indexes = []
    for i in range(1, files_num):
        filename = files_path + construct_filename(i, type_pattern)
        with open(filename) as file:
            try:
                msg = email.message_from_file(file).get_payload()
                if isinstance(msg, list):
                    msg = msg[0].get_payload()
                if isinstance(msg, list):
                    continue
                res.append(msg)
            except UnicodeDecodeError:
                continue
        indexes.append(i)
    return res, indexes


def extract_target(features_idxs, target_path=TARGET_PATH):
    with open(target_path, "r") as file:
        idxs = file.readlines()
    target = []
    for i in features_idxs:
        label = int(idxs[i][0])
        target.append(1.0) if label == 0 else target.append(-1.0)
    return target

if __name__ == '__main__':
    res, indexes = extract_data()
    target = extract_target(indexes)