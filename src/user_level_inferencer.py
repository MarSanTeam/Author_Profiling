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
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score
import logging
import itertools
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn import metrics

from configuration import BaseConfig
from data_prepration import prepare_ap_data, create_author_label
from data_loader import read_text, read_pickle, write_pickle
from utils import create_user_embedding, create_user_embedding_sbert, \
    create_user_embedding_irony
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

    DATA = TRAIN_DATA + VAL_DATA + TEST_DATA

    logging.debug("We have {} users in our data.".format(len(TRAIN_DATA) +
                                                         len(VAL_DATA) +
                                                         len(TEST_DATA)))

    # ----------------------- Indexer -----------------------
    TARGETS = [author_data[1] for author_data in DATA]

    TARGET_INDEXER = Indexer(vocabs=TARGETS)
    TARGET_INDEXER.build_vocab2idx()
    TARGET_INDEXER.save(path=CONFIG.assets_dir)

    logging.debug("Create Indexer")

    TARGETS_CONVENTIONAL = [[target] for target in TARGETS]
    INDEXED_TARGET = TARGET_INDEXER.convert_samples_to_indexes(TARGETS_CONVENTIONAL)
    INDEXED_TARGET = list(itertools.chain(*INDEXED_TARGET))

    logging.debug("Create indexed target")

    # create LM Tokenizer instance
    # TOKENIZER = BertTokenizer.from_pretrained(CONFIG.language_model_tokenizer_path)
    # MODEL = BertModel.from_pretrained(CONFIG.language_model_path, return_dict=True)
    # MODEL.eval()

    MODEL = SentenceTransformer(CONFIG.sentence_transformers_path, device="cuda:0")
    IRONY_MODEL = AutoModelForSequenceClassification.from_pretrained(
        CONFIG.roberta_base_irony_model_path)

    IRONY_TOKENIZER = AutoTokenizer.from_pretrained(CONFIG.roberta_base_irony_model_path)

    if os.path.exists(CONFIG.sbert_output_file_path):
        USER_EMBEDDINGS, USER_LABEL = read_pickle(CONFIG.sbert_output_file_path)
    else:
        USER_EMBEDDINGS, USER_LABEL = create_user_embedding_sbert(DATA, MODEL)  # , TOKENIZER)
        write_pickle(CONFIG.sbert_output_file_path, [USER_EMBEDDINGS, USER_LABEL])

    logging.debug("Create user embeddings")

    if os.path.exists(CONFIG.irony_output_file_path):
        USER_EMBEDDINGS_IRONY = read_pickle(CONFIG.irony_output_file_path)
    else:
        USER_EMBEDDINGS_IRONY, _ = create_user_embedding_irony(DATA, IRONY_MODEL, IRONY_TOKENIZER)
        write_pickle(CONFIG.irony_output_file_path, USER_EMBEDDINGS_IRONY)

    logging.debug("Create irony user embeddings")

    # ----------------------------- Train SVM -----------------------------
    FEATURES = list(np.concatenate([USER_EMBEDDINGS, USER_EMBEDDINGS_IRONY], axis=1))

    CLF = svm.SVC()
    # CLF.fit(FEATURES, INDEXED_TARGET)

    SCORES = cross_val_score(CLF, FEATURES, INDEXED_TARGET, cv=5)
    print(SCORES)

    print("%0.2f accuracy with a standard "
          "deviation of %0.2f" % (SCORES.mean(), SCORES.std()))
    # TRAIN_F1SCORE_MACRO = f1_score(TRAIN_INDEXED_TARGET, TRAIN_PREDICTED_TARGETS, average="macro")
    # VAL_F1SCORE_MACRO = f1_score(VAL_INDEXED_TARGET, VAL_PREDICTED_TARGETS, average="macro")
    # TEST_F1SCORE_MACRO = f1_score(TEST_INDEXED_TARGET, TEST_PREDICTED_TARGETS, average="macro")
    #
    # logging.debug(f"Train macro F1 score is : {TRAIN_F1SCORE_MACRO * 100:0.2f}")
    # logging.debug(f"Val macro F1 score is : {VAL_F1SCORE_MACRO * 100:0.2f}")
    # logging.debug(f"Test macro F1 score is : {TEST_F1SCORE_MACRO * 100:0.2f}")
    #
    # TRAIN_F1SCORE_MICRO = f1_score(TRAIN_INDEXED_TARGET, TRAIN_PREDICTED_TARGETS, average="micro")
    # VAL_F1SCORE_MICRO = f1_score(VAL_INDEXED_TARGET, VAL_PREDICTED_TARGETS, average="micro")
    # TEST_F1SCORE_MICRO = f1_score(TEST_INDEXED_TARGET, TEST_PREDICTED_TARGETS, average="micro")
    #
    # logging.debug(f"Train micro F1 score is : {TRAIN_F1SCORE_MICRO * 100:0.2f}")
    # logging.debug(f"Val micro F1 score is : {VAL_F1SCORE_MICRO * 100:0.2f}")
    # logging.debug(f"Test micro F1 score is : {TEST_F1SCORE_MICRO * 100:0.2f}")
