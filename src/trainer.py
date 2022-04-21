# -*- coding: utf-8 -*-
# ========================================================
"""trainer module is written for train model"""
# ========================================================


# ========================================================
# Imports
# ========================================================

import os
import logging
from sklearn.model_selection import train_test_split

from pytorch_lightning.loggers import CSVLogger
from transformers import T5Tokenizer, MT5Tokenizer

from configuration import BaseConfig
from data_prepration import create_author_label, prepare_ap_data
from data_loader import read_text

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    # create config instance
    CONFIG_CLASS = BaseConfig()
    CONFIG = CONFIG_CLASS.get_config()

    # create CSVLogger instance
    LOGGER = CSVLogger(save_dir=CONFIG.saved_model_path, name=CONFIG.model_name)

    # create LM Tokenizer instance
    # TOKENIZER = MT5Tokenizer.from_pretrained(CONFIG.language_model_tokenizer_path)

    DATA = read_text(path=os.path.join(CONFIG.raw_data_dir, CONFIG.truth_data))
    logging.debug("We have {} Author.".format(len(DATA)))

    AUTHOR2LABEL = create_author_label(DATA)

    DATA = prepare_ap_data(path=CONFIG.raw_data_dir, author2irony=AUTHOR2LABEL)

    TRAIN_DATA, TEST_DATA = train_test_split(DATA,
                                             test_size=0.3, random_state=1234)
    VAL_DATA, TEST_DATA = train_test_split(TEST_DATA,
                                           test_size=0.5, random_state=1234)

    logging.debug("We have {} authors in train data.".format(len(TRAIN_DATA)))
    logging.debug("We have {} authors in validation data.".format(len(VAL_DATA)))
    logging.debug("We have {} authors in test data.".format(len(TEST_DATA)))

    n_irony_author = 0
    n_not_irony_author = 0
    for data in TRAIN_DATA:
        if data[1] == "I":
            n_irony_author += 1
        else:
            n_not_irony_author += 1

    logging.debug("We have {} irony authors and {} not irony  authors in TRAIN_DATA.".format(
        n_irony_author, n_not_irony_author))
