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
from pytorch_lightning.callbacks import ModelCheckpoint

from configuration import BaseConfig
from data_prepration import create_author_label, prepare_ap_data
from data_loader import read_text, read_pickle, read_csv
from indexer import Indexer
from dataset import DataModule

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    # create config instance
    CONFIG_CLASS = BaseConfig()
    CONFIG = CONFIG_CLASS.get_config()

    # create CSVLogger instance
    LOGGER = CSVLogger(save_dir=CONFIG.saved_model_path, name=CONFIG.model_name)

    # create LM Tokenizer instance
    TOKENIZER = BertTokenizer.from_pretrained(CONFIG.language_model_tokenizer_path)

    TRAIN_DATA = read_csv(path=os.path.join(CONFIG.processed_data_dir,
                                            CONFIG.train_data),
                          columns=CONFIG.data_headers,
                          names=CONFIG.customized_headers)
    VAL_DATA = read_csv(path=os.path.join(CONFIG.processed_data_dir,
                                          CONFIG.val_data),
                        columns=CONFIG.data_headers,
                        names=CONFIG.customized_headers)
    TEST_DATA = read_csv(path=os.path.join(CONFIG.processed_data_dir,
                                           CONFIG.test_data),
                         columns=CONFIG.data_headers,
                         names=CONFIG.customized_headers)

    logging.debug("We have {} tweets in our data.".format(len(TRAIN_DATA) +
                                                          len(VAL_DATA) +
                                                          len(TEST_DATA)))

    logging.debug("We have {} tweets in train data.".format(len(TRAIN_DATA)))
    logging.debug("We have {} tweets in validation data.".format(len(VAL_DATA)))
    logging.debug("We have {} tweets in test data.".format(len(TEST_DATA)))

    # ----------------------- Indexer -----------------------
    TARGET_INDEXER = Indexer(vocabs=list(TRAIN_DATA.targets))
    TARGET_INDEXER.build_vocab2idx()
    TARGET_INDEXER.save(path=CONFIG.assets_dir)

    TRAIN_TARGETS_CONVENTIONAL = [[target] for target in list(TRAIN_DATA.targets)]
    TRAIN_INDEXED_TARGET = TARGET_INDEXER.convert_samples_to_indexes(TRAIN_TARGETS_CONVENTIONAL)

    VAL_TARGETS_CONVENTIONAL = [[target] for target in list(VAL_DATA.targets)]
    VAL_INDEXED_TARGET = TARGET_INDEXER.convert_samples_to_indexes(VAL_TARGETS_CONVENTIONAL)

    TEST_TARGETS_CONVENTIONAL = [[target] for target in list(TEST_DATA.targets)]
    TEST_INDEXED_TARGET = TARGET_INDEXER.convert_samples_to_indexes(TEST_TARGETS_CONVENTIONAL)

    # -------------------------------- Make DalaLoader Dict ----------------------------------------
    TRAIN_COLUMNS2DATA = {"texts": list(TRAIN_DATA.texts),
                          "targets": TRAIN_INDEXED_TARGET}
    VAL_COLUMNS2DATA = {"texts": list(VAL_DATA.texts),
                        "targets": VAL_INDEXED_TARGET}
    TEST_COLUMNS2DATA = {"texts": list(TEST_DATA.texts),
                         "targets": TRAIN_INDEXED_TARGET}

    DATA = {"train_data": TRAIN_COLUMNS2DATA,
            "val_data": VAL_COLUMNS2DATA, "test_data": TEST_COLUMNS2DATA}

    # ----------------------------- Create Data Module ----------------------------------
    DATA_MODULE = DataModule(data=DATA, config=CONFIG, tokenizer=TOKENIZER)
    DATA_MODULE.setup()
    CHECKPOINT_CALLBACK = ModelCheckpoint(monitor="val_loss",
                                          filename="QTag-{epoch:02d}-{val_loss:.2f}",
                                          save_top_k=CONFIG.save_top_k,
                                          mode="min")

    # a = []
    # for author in TRAIN_DATA:
    #     tweet_length = 0
    #     for tweet in author[0]:
    #         print(tweet)
    #         tweet_length += (len(TOKENIZER(tweet, return_tensors="pt").input_ids[0]))
    #     a.append(tweet_length)
    #
    # print(a)

    # tweets, labels = [], []
    # for author in TEST_DATA:
    #     for tweet in author[0]:
    #         tweets.append(tweet)
    #         labels.append(author[1])
    #
    # DATAFRAME = pd.DataFrame({"tweets": tweets, "labels": labels})
    # DATAFRAME = DATAFRAME.sample(frac=1).reset_index(drop=True)
    # DATAFRAME.to_csv("../data/Processed/test_tweet_level.csv", index=False)
