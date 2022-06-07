# -*- coding: utf-8 -*-
# ========================================================
"""trainer module is written for train model"""
# ========================================================


# ========================================================
# Imports
# ========================================================

import os
import pickle
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
from transformers import BertModel, BertTokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, T5Tokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score
import logging
import random
import itertools
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import accuracy_score

from configuration import BaseConfig
from data_prepration import prepare_ap_data, create_author_label
from data_loader import read_text, read_pickle, write_pickle
from utils import create_user_embedding, create_user_embedding_sbert, \
    create_user_embedding_irony, create_user_embedding_personality, \
    calculate_confidence_interval, cross_validator
from indexer import Indexer
from models.t5_personality import Classifier as personality_classofier
from models.t5_irony import Classifier as irony_classofier
from models.t5_emotion import Classifier as emotion_classofier

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

    # IRONY_MODEL = AutoModelForSequenceClassification.from_pretrained(
    #     CONFIG.roberta_base_irony_model_path)
    PERSONALITY_MODEL_PATH = "../assets/saved_models/personality/checkpoints/" \
                             "QTag-epoch=08-val_loss=0.65.ckpt"

    MYIRONY_MODEL_PATH = "../assets/saved_models/irony/checkpoints/" \
                         "QTag-epoch=10-val_loss=0.45.ckpt"

    EMOTION_MODEL_PATH = "../assets/saved_models/emotion/checkpoints/" \
                         "QTag-epoch=13-val_loss=0.45.ckpt"

    IRONY_TOKENIZER = AutoTokenizer.from_pretrained(CONFIG.roberta_base_irony_model_path)
    PERSONALITY_TOKENIZER = T5Tokenizer.from_pretrained(CONFIG.language_model_tokenizer_path)

    if os.path.exists(CONFIG.sbert_output_file_path):
        USER_EMBEDDINGS, USER_LABEL = read_pickle(CONFIG.sbert_output_file_path)
    else:
        MODEL = SentenceTransformer(CONFIG.sentence_transformers_path, device="cuda:0")
        MODEL = MODEL.to("cuda:0")

        USER_EMBEDDINGS, USER_LABEL = create_user_embedding_sbert(DATA, MODEL)  # , TOKENIZER)
        write_pickle(CONFIG.sbert_output_file_path, [USER_EMBEDDINGS, USER_LABEL])

    logging.debug("Create user embeddings")

    # if os.path.exists(CONFIG.irony_output_file_path):
    #     USER_EMBEDDINGS_IRONY = read_pickle(CONFIG.irony_output_file_path)
    # else:
    #     USER_EMBEDDINGS_IRONY, _ = create_user_embedding_irony(DATA, IRONY_MODEL, IRONY_TOKENIZER)
    #     write_pickle(CONFIG.irony_output_file_path, USER_EMBEDDINGS_IRONY)
    #
    # logging.debug("Create irony user embeddings")

    if os.path.exists(CONFIG.personality_output_file_path):
        USER_EMBEDDINGS_PERSONALITY = read_pickle(CONFIG.personality_output_file_path)
    else:
        PERSONALITY_MODEL = personality_classofier.load_from_checkpoint(PERSONALITY_MODEL_PATH, map_location="cuda:0")
        PERSONALITY_MODEL = PERSONALITY_MODEL.to("cuda:0")
        PERSONALITY_MODEL.eval()
        USER_EMBEDDINGS_PERSONALITY, _ = create_user_embedding_personality(DATA,
                                                                           PERSONALITY_MODEL,
                                                                           PERSONALITY_TOKENIZER,
                                                                           CONFIG.max_len)
        write_pickle(CONFIG.personality_output_file_path, USER_EMBEDDINGS_PERSONALITY)

    logging.debug("Create personality user embeddings")

    if os.path.exists(CONFIG.myirony_output_file_path):
        USER_EMBEDDINGS_MYIRONY = read_pickle(CONFIG.myirony_output_file_path)
    else:
        MYIRONY_MODEL = irony_classofier.load_from_checkpoint(MYIRONY_MODEL_PATH, map_location="cuda:0")
        MYIRONY_MODEL = MYIRONY_MODEL.to("cuda:0")
        MYIRONY_MODEL.eval()
        USER_EMBEDDINGS_MYIRONY, _ = create_user_embedding_personality(DATA,
                                                                       MYIRONY_MODEL,
                                                                       PERSONALITY_TOKENIZER,
                                                                       CONFIG.max_len)
        write_pickle(CONFIG.myirony_output_file_path, USER_EMBEDDINGS_MYIRONY)

    logging.debug("Create myirony user embeddings")

    if os.path.exists(CONFIG.emotion_output_file_path):
        USER_EMBEDDINGS_EMOTION = read_pickle(CONFIG.emotion_output_file_path)
    else:
        EMOTION_MODEL = irony_classofier.load_from_checkpoint(EMOTION_MODEL_PATH, map_location="cuda:0")
        EMOTION_MODEL = EMOTION_MODEL.to("cuda:0")

        EMOTION_MODEL.eval()
        USER_EMBEDDINGS_EMOTION, _ = create_user_embedding_personality(DATA,
                                                                       EMOTION_MODEL,
                                                                       PERSONALITY_TOKENIZER,
                                                                       CONFIG.max_len)
        write_pickle(CONFIG.emotion_output_file_path, USER_EMBEDDINGS_EMOTION)

    logging.debug("Create emotion user embeddings")

    # ----------------------------- Train SVM -----------------------------
    USER_EMBEDDINGS = np.squeeze(USER_EMBEDDINGS)
    print(np.shape(USER_EMBEDDINGS_MYIRONY))
    USER_EMBEDDINGS_MYIRONY = np.squeeze(USER_EMBEDDINGS_MYIRONY)
    USER_EMBEDDINGS_EMOTION = np.squeeze(USER_EMBEDDINGS_EMOTION)
    USER_EMBEDDINGS_PERSONALITY = np.squeeze(USER_EMBEDDINGS_PERSONALITY)

    FEATURES = list(np.concatenate([USER_EMBEDDINGS,
                                    USER_EMBEDDINGS_MYIRONY,
                                    USER_EMBEDDINGS_EMOTION,
                                    USER_EMBEDDINGS_PERSONALITY
                                    ], axis=1))
    # c = list(zip(FEATURES, INDEXED_TARGET))
    #
    # random.shuffle(c)
    #
    # FEATURES, INDEXED_TARGET = zip(*c)

    CLF = GradientBoostingClassifier()  # learning_rate=0.2, max_depth=3)#, n_estimators=50)
    # CLF = svm.SVC()
    CLF.fit(FEATURES, INDEXED_TARGET)

    # SCORES, CI = cross_validator(CLF, FEATURES, INDEXED_TARGET, cv=5)
    # print("%0.4f accuracy with a standard "
    #       "deviation of %0.4f" % (SCORES.mean(), SCORES.std()))
    # print("%0.4f ci with a standard "
    #       "deviation of %0.4f" % (CI.mean(), CI.std()))
    # SCORES, CI = cross_validator(CLF, FEATURES, INDEXED_TARGET, cv=10)
    # print("%0.4f accuracy with a standard "
    #       "deviation of %0.4f" % (SCORES.mean(), SCORES.std()))
    # print("%0.4f ci with a standard "
    #       "deviation of %0.4f" % (CI.mean(), CI.std()))
# SCORES = cross_val_score(CLF, FEATURES, INDEXED_TARGET, cv=5)
# print(SCORES)
# print(CI)
# CI = 0
# for s in SCORES:
#     CI += calculate_confidence_interval(s, len(FEATURES)//5, 95)
# print(CI/5)
#

    # y_true = CLF.predict(FEATURES[-80:])
    # print(accuracy_score(y_true, INDEXED_TARGET[-80:]))
    filename = "finalized_model_new.sav"
    pickle.dump(CLF, open(filename, "wb"))

    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.predict(FEATURES[-80:])#, INDEXED_TARGET[-80:])
    print(result)
    print(INDEXED_TARGET[-80:])
    print(accuracy_score(result, INDEXED_TARGET[-80:]))

# print(result)

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
