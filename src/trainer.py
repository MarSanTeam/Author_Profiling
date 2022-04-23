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
from data_loader import read_text, read_pickle

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    # create config instance
    CONFIG_CLASS = BaseConfig()
    CONFIG = CONFIG_CLASS.get_config()

    # create CSVLogger instance
    LOGGER = CSVLogger(save_dir=CONFIG.saved_model_path, name=CONFIG.model_name)

    TRAIN_DATA = read_pickle(path=os.path.join(CONFIG.processed_data_dir,
                                               CONFIG.train_data))
    VAL_DATA = read_pickle(path=os.path.join(CONFIG.processed_data_dir,
                                             CONFIG.val_data))
    TEST_DATA = read_pickle(path=os.path.join(CONFIG.processed_data_dir,
                                              CONFIG.test_data))

    # create LM Tokenizer instance
    # TOKENIZER = MT5Tokenizer.from_pretrained(CONFIG.language_model_tokenizer_path)
