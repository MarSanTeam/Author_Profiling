# -*- coding: utf-8 -*-
# ========================================================

"""
    Author Profiling Project:
        src:
                user_level_trainer.py
"""

# ============================ Third Party libs ============================
import os
import random
import pickle
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
from transformers import T5Tokenizer
from sentence_transformers import SentenceTransformer
import logging
import itertools
from sklearn.metrics import accuracy_score

# ============================ My packages ============================
from configuration import BaseConfig
from data_loader import read_pickle, write_pickle
from utils import create_sbert_user_embedding, create_user_embedding, cross_validator, \
    calculate_confidence_interval
from indexer import Indexer
from models.t5_personality import Classifier as PersonalityClassifier
from models.t5_irony import Classifier as IronyClassifier

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    # ---------------------- create config instance --------------------------------
    CONFIG_CLASS = BaseConfig()
    CONFIG = CONFIG_CLASS.get_config()

    PERSONALITY_MODEL_PATH = os.path.join(
        CONFIG.assets_dir, "personality/checkpoints/QTag-epoch=08-val_loss=0.65.ckpt")

    IRONY_MODEL_PATH = os.path.join(
        CONFIG.assets_dir, "irony/checkpoints/QTag-epoch=10-val_loss=0.45.ckpt")

    EMOTION_MODEL_PATH = os.path.join(
        CONFIG.assets_dir, "emotion/checkpoints/QTag-epoch=13-val_loss=0.45.ckpt")

    # ------------------------------ create LM Tokenizer instance-------------------
    TOKENIZER = T5Tokenizer.from_pretrained(CONFIG.language_model_path)

    # ---------------------------------- Loading data ------------------------------

    TRAIN_DATA = read_pickle(os.path.join(CONFIG.processed_data_dir, "train_data.pkl"))
    VAL_DATA = read_pickle(os.path.join(CONFIG.processed_data_dir, "val_data.pkl"))
    TEST_DATA = read_pickle(os.path.join(CONFIG.processed_data_dir, "test_data.pkl"))

    DATA = TRAIN_DATA + VAL_DATA + TEST_DATA

    logging.debug("We have {} users in our data.".format(len(TRAIN_DATA) +
                                                         len(VAL_DATA) +
                                                         len(TEST_DATA)))

    # ---------------------------------- Indexer ---------------------------------
    TARGETS = [author_data[1] for author_data in DATA]

    TARGET_INDEXER = Indexer(vocabs=TARGETS)
    TARGET_INDEXER.build_vocab2idx()
    TARGET_INDEXER.save(path=CONFIG.assets_dir)

    TARGETS_CONVENTIONAL = [[target] for target in TARGETS]
    INDEXED_TARGET = TARGET_INDEXER.convert_samples_to_indexes(TARGETS_CONVENTIONAL)
    INDEXED_TARGET = list(itertools.chain(*INDEXED_TARGET))

    logging.debug("Create indexed target")

    if os.path.exists(CONFIG.sbert_output_file_path):
        logging.debug("Load Sbert user embeddings")
        SBERT_USER_EMBEDDINGS, USER_LABEL = read_pickle(CONFIG.sbert_output_file_path)
    else:
        logging.debug("Create Sbert user embeddings")
        MODEL = SentenceTransformer(CONFIG.sentence_transformers_path, device=CONFIG.device)
        SBERT_USER_EMBEDDINGS, USER_LABEL = create_sbert_user_embedding(DATA, MODEL)
        write_pickle(CONFIG.sbert_output_file_path, [SBERT_USER_EMBEDDINGS, USER_LABEL])

    if os.path.exists(CONFIG.personality_output_file_path):
        logging.debug("Load personality user embeddings")
        PERSONALITY_USER_EMBEDDINGS = read_pickle(CONFIG.personality_output_file_path)
    else:
        logging.debug("Create personality user embeddings")
        PERSONALITY_MODEL = PersonalityClassifier.load_from_checkpoint(PERSONALITY_MODEL_PATH,
                                                                       map_location=CONFIG.device)
        PERSONALITY_MODEL.eval()
        PERSONALITY_USER_EMBEDDINGS, _ = create_user_embedding(DATA,
                                                               PERSONALITY_MODEL,
                                                               TOKENIZER,
                                                               CONFIG.max_len)
        write_pickle(CONFIG.personality_output_file_path, PERSONALITY_USER_EMBEDDINGS)

    if os.path.exists(CONFIG.irony_output_file_path):
        logging.debug("Load irony user embeddings")
        IRONY_USER_EMBEDDINGS = read_pickle(CONFIG.myirony_output_file_path)
    else:
        logging.debug("Create irony user embeddings")
        IRONY_MODEL = IronyClassifier.load_from_checkpoint(IRONY_MODEL_PATH,
                                                           map_location=CONFIG.device)
        IRONY_MODEL.eval()
        IRONY_USER_EMBEDDINGS, _ = create_user_embedding(DATA,
                                                         IRONY_MODEL,
                                                         TOKENIZER,
                                                         CONFIG.max_len)
        write_pickle(CONFIG.irony_output_file_path, IRONY_USER_EMBEDDINGS)

    if os.path.exists(CONFIG.emotion_output_file_path):
        logging.debug("Load emotion user embeddings")
        EMOTION_USER_EMBEDDINGS = read_pickle(CONFIG.emotion_output_file_path)
    else:
        logging.debug("Create emotion user embeddings")
        EMOTION_MODEL = IronyClassifier.load_from_checkpoint(EMOTION_MODEL_PATH,
                                                             map_location=CONFIG.device)
        EMOTION_MODEL.eval()
        EMOTION_USER_EMBEDDINGS, _ = create_user_embedding(DATA,
                                                           EMOTION_MODEL,
                                                           TOKENIZER,
                                                           CONFIG.max_len)
        write_pickle(CONFIG.emotion_output_file_path, EMOTION_USER_EMBEDDINGS)

    # ----------------------------- Train SVM -----------------------------
    SBERT_USER_EMBEDDINGS = np.squeeze(SBERT_USER_EMBEDDINGS)
    IRONY_USER_EMBEDDINGS = np.squeeze(IRONY_USER_EMBEDDINGS)
    EMOTION_USER_EMBEDDINGS = np.squeeze(EMOTION_USER_EMBEDDINGS)
    PERSONALITY_USER_EMBEDDINGS = np.squeeze(PERSONALITY_USER_EMBEDDINGS)

    FEATURES = list(np.concatenate([SBERT_USER_EMBEDDINGS,
                                    IRONY_USER_EMBEDDINGS,
                                    EMOTION_USER_EMBEDDINGS,
                                    PERSONALITY_USER_EMBEDDINGS],
                                   axis=1))
    c = list(zip(FEATURES, INDEXED_TARGET))
    random.shuffle(c)
    FEATURES, INDEXED_TARGET = zip(*c)

    CLF = GradientBoostingClassifier()
    CLF.fit(FEATURES, INDEXED_TARGET)

    SCORES, CI = cross_validator(CLF, FEATURES, INDEXED_TARGET, cv=5)
    logging.debug(
        "%0.4f accuracy with a standard deviation of %0.4f" % (SCORES.mean(), SCORES.std()))
    logging.debug("%0.4f ci with a standard deviation of %0.4f" % (CI.mean(), CI.std()))
    SCORES, CI = cross_validator(CLF, FEATURES, INDEXED_TARGET, cv=10)
    logging.debug(
        "%0.4f accuracy with a standard deviation of %0.4f" % (SCORES.mean(), SCORES.std()))
    logging.debug("%0.4f ci with a standard deviation of %0.4f" % (CI.mean(), CI.std()))
