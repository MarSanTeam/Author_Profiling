# -*- coding: utf-8 -*-
# ========================================================
"""trainer module is written for train model"""
# ========================================================


# ========================================================
# Imports
# ========================================================

import os
from sklearn import svm
import numpy as np
from transformers import BertModel, BertTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score
import logging
import itertools

from configuration import BaseConfig
from data_prepration import prepare_ap_data, create_author_label
from data_loader import read_text, read_pickle
from utils import create_user_embedding, create_user_embedding_sbert
from indexer import Indexer

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    # create config instance
    CONFIG_CLASS = BaseConfig()
    CONFIG = CONFIG_CLASS.get_config()

    logging.debug("starts")

    TRAIN_DATA = read_pickle(os.path.join(CONFIG.processed_data_dir, "train_data.pkl"))
    VAL_DATA = read_pickle(os.path.join(CONFIG.processed_data_dir, "val_data.pkl"))
    TEST_DATA = read_pickle(os.path.join(CONFIG.processed_data_dir, "test_data.pkl"))

    logging.debug("We have {} users in our data.".format(len(TRAIN_DATA) +
                                                         len(VAL_DATA) +
                                                         len(TEST_DATA)))

    logging.debug("We have {} users in train data.".format(len(TRAIN_DATA)))
    logging.debug("We have {} users in validation data.".format(len(VAL_DATA)))
    logging.debug("We have {} users in test data.".format(len(TEST_DATA)))

    # ----------------------- Indexer -----------------------
    TRAIN_TARGETS = [author_data[1] for author_data in TRAIN_DATA]
    VAL_TARGETS = [author_data[1] for author_data in VAL_DATA]
    TEST_TARGETS = [author_data[1] for author_data in TEST_DATA]

    TARGET_INDEXER = Indexer(vocabs=TRAIN_TARGETS)
    TARGET_INDEXER.build_vocab2idx()
    TARGET_INDEXER.save(path=CONFIG.assets_dir)

    logging.debug("Create Indexer")

    TRAIN_TARGETS_CONVENTIONAL = [[target] for target in TRAIN_TARGETS]
    TRAIN_INDEXED_TARGET = TARGET_INDEXER.convert_samples_to_indexes(TRAIN_TARGETS_CONVENTIONAL)
    TRAIN_INDEXED_TARGET = list(itertools.chain(*TRAIN_INDEXED_TARGET))

    logging.debug("Create train indexed target")

    VAL_TARGETS_CONVENTIONAL = [[target] for target in VAL_TARGETS]
    VAL_INDEXED_TARGET = TARGET_INDEXER.convert_samples_to_indexes(VAL_TARGETS_CONVENTIONAL)
    VAL_INDEXED_TARGET = list(itertools.chain(*VAL_INDEXED_TARGET))

    logging.debug("Create validation indexed target")

    TEST_TARGETS_CONVENTIONAL = [[target] for target in TEST_TARGETS]
    TEST_INDEXED_TARGET = TARGET_INDEXER.convert_samples_to_indexes(TEST_TARGETS_CONVENTIONAL)
    TEST_INDEXED_TARGET = list(itertools.chain(*TEST_INDEXED_TARGET))

    logging.debug("Create test indexed target")

    # create LM Tokenizer instance
    # TOKENIZER = BertTokenizer.from_pretrained(CONFIG.language_model_tokenizer_path)
    # MODEL = BertModel.from_pretrained(CONFIG.language_model_path, return_dict=True)
    # MODEL.eval()

    MODEL = SentenceTransformer(CONFIG.sentence_transformers_path, device="cuda:0")

    TRAIN_USER_EMBEDDINGS, TRAIN_USER_LABEL = create_user_embedding_sbert(TRAIN_DATA, MODEL)  # , TOKENIZER)
    logging.debug("Create train user embeddings")

    VAL_USER_EMBEDDINGS, VAL_USER_LABEL = create_user_embedding_sbert(VAL_DATA, MODEL)  # , TOKENIZER)
    logging.debug("Create validation user embeddings")

    TEST_USER_EMBEDDINGS, TEST_USER_LABEL = create_user_embedding_sbert(TEST_DATA, MODEL)  # , TOKENIZER)
    logging.debug("Create test user embeddings")

    # ----------------------------- Train SVM -----------------------------
    CLF = svm.SVC()
    CLF.fit(TRAIN_USER_EMBEDDINGS, TRAIN_INDEXED_TARGET)

    TRAIN_PREDICTED_TARGETS = CLF.predict(TRAIN_USER_EMBEDDINGS)
    VAL_PREDICTED_TARGETS = CLF.predict(VAL_USER_EMBEDDINGS)
    TEST_PREDICTED_TARGETS = CLF.predict(TEST_USER_EMBEDDINGS)

    TRAIN_F1SCORE_MACRO = f1_score(TRAIN_INDEXED_TARGET, TRAIN_PREDICTED_TARGETS, average="macro")
    VAL_F1SCORE_MACRO = f1_score(VAL_INDEXED_TARGET, VAL_PREDICTED_TARGETS, average="macro")
    TEST_F1SCORE_MACRO = f1_score(TEST_INDEXED_TARGET, TEST_PREDICTED_TARGETS, average="macro")

    logging.debug(f"Train macro F1 score is : {TRAIN_F1SCORE_MACRO * 100:0.2f}")
    logging.debug(f"Val macro F1 score is : {VAL_F1SCORE_MACRO * 100:0.2f}")
    logging.debug(f"Test macro F1 score is : {TEST_F1SCORE_MACRO * 100:0.2f}")

    TRAIN_F1SCORE_MICRO = f1_score(TRAIN_INDEXED_TARGET, TRAIN_PREDICTED_TARGETS, average="micro")
    VAL_F1SCORE_MICRO = f1_score(VAL_INDEXED_TARGET, VAL_PREDICTED_TARGETS, average="micro")
    TEST_F1SCORE_MICRO = f1_score(TEST_INDEXED_TARGET, TEST_PREDICTED_TARGETS, average="micro")

    logging.debug(f"Train micro F1 score is : {TRAIN_F1SCORE_MICRO * 100:0.2f}")
    logging.debug(f"Val micro F1 score is : {VAL_F1SCORE_MICRO * 100:0.2f}")
    logging.debug(f"Test micro F1 score is : {TEST_F1SCORE_MICRO * 100:0.2f}")
