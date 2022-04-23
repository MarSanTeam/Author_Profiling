# -*- coding: utf-8 -*-
# ========================================================
"""trainer module is written for train model"""
# ========================================================


# ========================================================
# Imports
# ========================================================

import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from pytorch_lightning.loggers import CSVLogger
from transformers import T5Tokenizer, BertTokenizer

from configuration import BaseConfig
from data_prepration import create_author_label, prepare_ap_data
from data_loader import read_text, read_pickle, read_csv

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    # create config instance
    CONFIG_CLASS = BaseConfig()
    CONFIG = CONFIG_CLASS.get_config()

    # create CSVLogger instance
    LOGGER = CSVLogger(save_dir=CONFIG.saved_model_path, name=CONFIG.model_name)

    TRAIN_DATA = read_csv(path=os.path.join(CONFIG.processed_data_dir,
                                            CONFIG.train_data))
    VAL_DATA = read_csv(path=os.path.join(CONFIG.processed_data_dir,
                                          CONFIG.val_data))
    TEST_DATA = read_csv(path=os.path.join(CONFIG.processed_data_dir,
                                           CONFIG.test_data))

    logging.debug("We have {} authors in our data.".format(len(TRAIN_DATA) +
                                                           len(VAL_DATA) +
                                                           len(TEST_DATA)))

    logging.debug("We have {} authors in train data.".format(len(TRAIN_DATA)))
    logging.debug("We have {} authors in validation data.".format(len(VAL_DATA)))
    logging.debug("We have {} authors in test data.".format(len(TEST_DATA)))

    # create LM Tokenizer instance
    TOKENIZER = BertTokenizer.from_pretrained(CONFIG.language_model_tokenizer_path)

    # a = []
    # for author in TRAIN_DATA:
    #     tweet_length = 0
    #     for tweet in author[0]:
    #         print(tweet)
    #         tweet_length += (len(TOKENIZER(tweet, return_tensors="pt").input_ids[0]))
    #     a.append(tweet_length)
    #
    # print(a)
# -*- coding: utf-8 -*-
# ========================================================
"""trainer module is written for train model"""
# ========================================================


# ========================================================
# Imports
# ========================================================

import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from pytorch_lightning.loggers import CSVLogger
from transformers import T5Tokenizer, BertTokenizer

from configuration import BaseConfig
from data_prepration import create_author_label, prepare_ap_data
from data_loader import read_text, read_pickle, read_csv

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    # create config instance
    CONFIG_CLASS = BaseConfig()
    CONFIG = CONFIG_CLASS.get_config()

    # create CSVLogger instance
    LOGGER = CSVLogger(save_dir=CONFIG.saved_model_path, name=CONFIG.model_name)

    TRAIN_DATA = read_csv(path=os.path.join(CONFIG.processed_data_dir,
                                            CONFIG.train_data))
    VAL_DATA = read_csv(path=os.path.join(CONFIG.processed_data_dir,
                                          CONFIG.val_data))
    TEST_DATA = read_csv(path=os.path.join(CONFIG.processed_data_dir,
                                           CONFIG.test_data))

    logging.debug("We have {} authors in our data.".format(len(TRAIN_DATA) +
                                                           len(VAL_DATA) +
                                                           len(TEST_DATA)))

    logging.debug("We have {} authors in train data.".format(len(TRAIN_DATA)))
    logging.debug("We have {} authors in validation data.".format(len(VAL_DATA)))
    logging.debug("We have {} authors in test data.".format(len(TEST_DATA)))

    # create LM Tokenizer instance
    TOKENIZER = BertTokenizer.from_pretrained(CONFIG.language_model_tokenizer_path)

    # a = []
    # for author in TRAIN_DATA:
    #     tweet_length = 0
    #     for tweet in author[0]:
    #         print(tweet)
    #         tweet_length += (len(TOKENIZER(tweet, return_tensors="pt").input_ids[0]))
    #     a.append(tweet_length)
    #
    # print(a)
# -*- coding: utf-8 -*-
# ========================================================
"""trainer module is written for train model"""
# ========================================================


# ========================================================
# Imports
# ========================================================

import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from pytorch_lightning.loggers import CSVLogger
from transformers import T5Tokenizer, BertTokenizer

from configuration import BaseConfig
from data_prepration import create_author_label, prepare_ap_data
from data_loader import read_text, read_pickle, read_csv

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    # create config instance
    CONFIG_CLASS = BaseConfig()
    CONFIG = CONFIG_CLASS.get_config()

    # create CSVLogger instance
    LOGGER = CSVLogger(save_dir=CONFIG.saved_model_path, name=CONFIG.model_name)

    TRAIN_DATA = read_csv(path=os.path.join(CONFIG.processed_data_dir,
                                            CONFIG.train_data))
    VAL_DATA = read_csv(path=os.path.join(CONFIG.processed_data_dir,
                                          CONFIG.val_data))
    TEST_DATA = read_csv(path=os.path.join(CONFIG.processed_data_dir,
                                           CONFIG.test_data))

    logging.debug("We have {} authors in our data.".format(len(TRAIN_DATA) +
                                                           len(VAL_DATA) +
                                                           len(TEST_DATA)))

    logging.debug("We have {} authors in train data.".format(len(TRAIN_DATA)))
    logging.debug("We have {} authors in validation data.".format(len(VAL_DATA)))
    logging.debug("We have {} authors in test data.".format(len(TEST_DATA)))

    # create LM Tokenizer instance
    TOKENIZER = BertTokenizer.from_pretrained(CONFIG.language_model_tokenizer_path)

    # a = []
    # for author in TRAIN_DATA:
    #     tweet_length = 0
    #     for tweet in author[0]:
    #         print(tweet)
    #         tweet_length += (len(TOKENIZER(tweet, return_tensors="pt").input_ids[0]))
    #     a.append(tweet_length)
    #
    # print(a)
# -*- coding: utf-8 -*-
# ========================================================
"""trainer module is written for train model"""
# ========================================================


# ========================================================
# Imports
# ========================================================

import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from pytorch_lightning.loggers import CSVLogger
from transformers import T5Tokenizer, BertTokenizer

from configuration import BaseConfig
from data_prepration import create_author_label, prepare_ap_data
from data_loader import read_text, read_pickle, read_csv

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    # create config instance
    CONFIG_CLASS = BaseConfig()
    CONFIG = CONFIG_CLASS.get_config()

    # create CSVLogger instance
    LOGGER = CSVLogger(save_dir=CONFIG.saved_model_path, name=CONFIG.model_name)

    TRAIN_DATA = read_csv(path=os.path.join(CONFIG.processed_data_dir,
                                            CONFIG.train_data))
    VAL_DATA = read_csv(path=os.path.join(CONFIG.processed_data_dir,
                                          CONFIG.val_data))
    TEST_DATA = read_csv(path=os.path.join(CONFIG.processed_data_dir,
                                           CONFIG.test_data))

    logging.debug("We have {} authors in our data.".format(len(TRAIN_DATA) +
                                                           len(VAL_DATA) +
                                                           len(TEST_DATA)))

    logging.debug("We have {} authors in train data.".format(len(TRAIN_DATA)))
    logging.debug("We have {} authors in validation data.".format(len(VAL_DATA)))
    logging.debug("We have {} authors in test data.".format(len(TEST_DATA)))

    # create LM Tokenizer instance
    TOKENIZER = BertTokenizer.from_pretrained(CONFIG.language_model_tokenizer_path)

    # a = []
    # for author in TRAIN_DATA:
    #     tweet_length = 0
    #     for tweet in author[0]:
    #         print(tweet)
    #         tweet_length += (len(TOKENIZER(tweet, return_tensors="pt").input_ids[0]))
    #     a.append(tweet_length)
    #
    # print(a)
