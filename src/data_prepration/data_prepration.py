from typing import List

from data_loader import read_xml
import os


def prepare_ap_data(path: str, author2irony: dict) -> list:
    """

    :param path:
    :param author2irony:
    :return:
    """
    data = []
    for file in os.listdir(path):
        if file[-4:] == ".xml":
            author_tweets = []
            author_id = file[:-4]
            root = read_xml(os.path.join(path, file))
            for child in root:
                for inch in child:
                    author_tweets.append(inch.text)
            data.append([author_tweets, author2irony[author_id]])
    return data


# if name == "__main__":
#     DATA = read_text(path=os.path.join(CONFIG.raw_data_dir, CONFIG.truth_data))
#     logging.debug("We have {} Author.".format(len(DATA)))
#
#     AUTHOR2LABEL = create_author_label(DATA)
#
#     DATA = prepare_ap_data(path=CONFIG.raw_data_dir, author2irony=AUTHOR2LABEL)
#
#     TRAIN_DATA, TEST_DATA = train_test_split(DATA,
#                                              test_size=0.3, random_state=1234)
#     VAL_DATA, TEST_DATA = train_test_split(TEST_DATA,
#                                            test_size=0.5, random_state=1234)
#
#     logging.debug("We have {} authors in train data.".format(len(TRAIN_DATA)))
#     logging.debug("We have {} authors in validation data.".format(len(VAL_DATA)))
#     logging.debug("We have {} authors in test data.".format(len(TEST_DATA)))
#
#     write_pickle(path=os.path.join(CONFIG.processed_data_dir, "train_data.pkl"),
#                  data=TRAIN_DATA)
#     write_pickle(path=os.path.join(CONFIG.processed_data_dir, "val_data.pkl"),
#                  data=VAL_DATA)
#     write_pickle(path=os.path.join(CONFIG.processed_data_dir, "test_data.pkl"),
#                  data=TEST_DATA)
