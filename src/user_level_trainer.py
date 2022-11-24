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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
import numpy as np
from transformers import T5Tokenizer
from sentence_transformers import SentenceTransformer
import logging
import itertools

# ============================ My packages ============================
from configuration import BaseConfig
from data_loader import read_pickle, write_pickle
from utils import create_sbert_user_embedding, create_user_embedding
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

    # ------------------------------ create User Embeddings -------------------
    if os.path.exists(CONFIG.sbert_output_file_path):
        logging.debug("Load Sbert user embeddings")
        TRAIN_DATA, VAL_DATA, TEST_DATA = read_pickle(CONFIG.sbert_output_file_path)
        TRAIN_SBERT_USER_EMBEDDINGS, TRAIN_USER_LABEL = TRAIN_DATA[0], TRAIN_DATA[1]
        VAL_SBERT_USER_EMBEDDINGS, VAL_USER_LABEL = VAL_DATA[0], VAL_DATA[1]
        TEST_SBERT_USER_EMBEDDINGS, TEST_USER_LABEL = TEST_DATA[0], TEST_DATA[1]
    else:
        logging.debug("Create Sbert user embeddings")
        MODEL = SentenceTransformer(CONFIG.sentence_transformers_path, device=CONFIG.device)

        TRAIN_SBERT_USER_EMBEDDINGS, TRAIN_USER_LABEL = create_sbert_user_embedding(TRAIN_DATA,
                                                                                    MODEL)
        logging.debug("Create train user embeddings")

        VAL_SBERT_USER_EMBEDDINGS, VAL_USER_LABEL = create_sbert_user_embedding(VAL_DATA, MODEL)
        logging.debug("Create validation user embeddings")

        TEST_SBERT_USER_EMBEDDINGS, TEST_USER_LABEL = create_sbert_user_embedding(TEST_DATA, MODEL)
        logging.debug("Create test user embeddings")

        write_pickle(CONFIG.sbert_output_file_path,
                     [[TRAIN_SBERT_USER_EMBEDDINGS, TRAIN_USER_LABEL],
                      [VAL_SBERT_USER_EMBEDDINGS, VAL_USER_LABEL],
                      [TEST_SBERT_USER_EMBEDDINGS, TEST_USER_LABEL]])

    if os.path.exists(CONFIG.personality_output_file_path):
        logging.debug("Load personality user embeddings")
        TRAIN_PERSONALITY_USER_EMBEDDINGS, VAL_PERSONALITY_USER_EMBEDDINGS, \
        TEST_PERSONALITY_USER_EMBEDDINGS = read_pickle(CONFIG.personality_output_file_path)
    else:
        logging.debug("Create personality user embeddings")
        PERSONALITY_MODEL = PersonalityClassifier.load_from_checkpoint(PERSONALITY_MODEL_PATH,
                                                                       map_location=CONFIG.device)
        PERSONALITY_MODEL.eval()
        TRAIN_PERSONALITY_USER_EMBEDDINGS, _ = create_user_embedding(TRAIN_DATA,
                                                                     PERSONALITY_MODEL,
                                                                     TOKENIZER,
                                                                     CONFIG.max_len)
        VAL_PERSONALITY_USER_EMBEDDINGS, _ = create_user_embedding(VAL_DATA,
                                                                   PERSONALITY_MODEL,
                                                                   TOKENIZER,
                                                                   CONFIG.max_len)
        TEST_PERSONALITY_USER_EMBEDDINGS, _ = create_user_embedding(TEST_DATA,
                                                                    PERSONALITY_MODEL,
                                                                    TOKENIZER,
                                                                    CONFIG.max_len)
        write_pickle(CONFIG.personality_output_file_path,
                     [TRAIN_PERSONALITY_USER_EMBEDDINGS, VAL_PERSONALITY_USER_EMBEDDINGS,
                      TEST_PERSONALITY_USER_EMBEDDINGS])

    if os.path.exists(CONFIG.irony_output_file_path):
        logging.debug("Load irony user embeddings")
        TRAIN_IRONY_USER_EMBEDDINGS, VAL_IRONY_USER_EMBEDDINGS, TEST_IRONY_USER_EMBEDDINGS \
            = read_pickle(CONFIG.myirony_output_file_path)
    else:
        logging.debug("Create irony user embeddings")
        IRONY_MODEL = IronyClassifier.load_from_checkpoint(IRONY_MODEL_PATH,
                                                           map_location=CONFIG.device)
        IRONY_MODEL.eval()
        TRAIN_IRONY_USER_EMBEDDINGS, _ = create_user_embedding(TRAIN_DATA,
                                                               IRONY_MODEL,
                                                               TOKENIZER,
                                                               CONFIG.max_len)
        VAL_IRONY_USER_EMBEDDINGS, _ = create_user_embedding(VAL_DATA,
                                                             IRONY_MODEL,
                                                             TOKENIZER,
                                                             CONFIG.max_len)
        TEST_IRONY_USER_EMBEDDINGS, _ = create_user_embedding(TEST_DATA,
                                                              IRONY_MODEL,
                                                              TOKENIZER,
                                                              CONFIG.max_len)
        write_pickle(CONFIG.irony_output_file_path,
                     [TRAIN_IRONY_USER_EMBEDDINGS, VAL_IRONY_USER_EMBEDDINGS,
                      TEST_IRONY_USER_EMBEDDINGS])

    if os.path.exists(CONFIG.emotion_output_file_path):
        logging.debug("Load emotion user embeddings")
        TRAIN_EMOTION_USER_EMBEDDINGS, VAL_EMOTION_USER_EMBEDDINGS, TEST_EMOTION_USER_EMBEDDINGS \
            = read_pickle(CONFIG.emotion_output_file_path)
    else:
        logging.debug("Create emotion user embeddings")
        EMOTION_MODEL = IronyClassifier.load_from_checkpoint(EMOTION_MODEL_PATH,
                                                             map_location=CONFIG.device)
        EMOTION_MODEL.eval()
        TRAIN_EMOTION_USER_EMBEDDINGS, _ = create_user_embedding(TRAIN_DATA,
                                                                 EMOTION_MODEL,
                                                                 TOKENIZER,
                                                                 CONFIG.max_len)
        VAL_EMOTION_USER_EMBEDDINGS, _ = create_user_embedding(VAL_DATA,
                                                               EMOTION_MODEL,
                                                               TOKENIZER,
                                                               CONFIG.max_len)
        TEST_EMOTION_USER_EMBEDDINGS, _ = create_user_embedding(TEST_DATA,
                                                                EMOTION_MODEL,
                                                                TOKENIZER,
                                                                CONFIG.max_len)
        write_pickle(CONFIG.emotion_output_file_path,
                     [TRAIN_EMOTION_USER_EMBEDDINGS, VAL_EMOTION_USER_EMBEDDINGS,
                      TEST_EMOTION_USER_EMBEDDINGS])

    # ----------------------------- Train SVM -----------------------------
    TRAIN_SBERT_USER_EMBEDDINGS = np.squeeze(TRAIN_SBERT_USER_EMBEDDINGS)
    TRAIN_IRONY_USER_EMBEDDINGS = np.squeeze(TRAIN_IRONY_USER_EMBEDDINGS)
    TRAIN_EMOTION_USER_EMBEDDINGS = np.squeeze(TRAIN_EMOTION_USER_EMBEDDINGS)
    TRAIN_PERSONALITY_USER_EMBEDDINGS = np.squeeze(TRAIN_PERSONALITY_USER_EMBEDDINGS)

    VAL_SBERT_USER_EMBEDDINGS = np.squeeze(VAL_SBERT_USER_EMBEDDINGS)
    VAL_IRONY_USER_EMBEDDINGS = np.squeeze(VAL_IRONY_USER_EMBEDDINGS)
    VAL_EMOTION_USER_EMBEDDINGS = np.squeeze(VAL_EMOTION_USER_EMBEDDINGS)
    VAL_PERSONALITY_USER_EMBEDDINGS = np.squeeze(VAL_PERSONALITY_USER_EMBEDDINGS)

    TEST_SBERT_USER_EMBEDDINGS = np.squeeze(TEST_SBERT_USER_EMBEDDINGS)
    TEST_IRONY_USER_EMBEDDINGS = np.squeeze(TEST_IRONY_USER_EMBEDDINGS)
    TEST_EMOTION_USER_EMBEDDINGS = np.squeeze(TEST_EMOTION_USER_EMBEDDINGS)
    TEST_PERSONALITY_USER_EMBEDDINGS = np.squeeze(TEST_PERSONALITY_USER_EMBEDDINGS)

    TRAIN_FEATURES = list(np.concatenate([TRAIN_SBERT_USER_EMBEDDINGS,
                                          TRAIN_IRONY_USER_EMBEDDINGS,
                                          TRAIN_EMOTION_USER_EMBEDDINGS,
                                          TRAIN_PERSONALITY_USER_EMBEDDINGS],
                                         axis=1))

    VAL_FEATURES = list(np.concatenate([VAL_SBERT_USER_EMBEDDINGS,
                                        VAL_IRONY_USER_EMBEDDINGS,
                                        VAL_EMOTION_USER_EMBEDDINGS,
                                        VAL_PERSONALITY_USER_EMBEDDINGS],
                                       axis=1))

    TEST_FEATURES = list(np.concatenate([TEST_SBERT_USER_EMBEDDINGS,
                                         TEST_IRONY_USER_EMBEDDINGS,
                                         TEST_EMOTION_USER_EMBEDDINGS,
                                         TEST_PERSONALITY_USER_EMBEDDINGS],
                                        axis=1))

    c = list(zip(TRAIN_FEATURES, TRAIN_INDEXED_TARGET))
    random.shuffle(c)
    TRAIN_FEATURES, TRAIN_INDEXED_TARGET = zip(*c)

    CLF = GradientBoostingClassifier()
    CLF.fit(TRAIN_FEATURES, TRAIN_INDEXED_TARGET)

    TRAIN_PREDICTED_TARGETS = CLF.predict(TRAIN_FEATURES)
    VAL_PREDICTED_TARGETS = CLF.predict(VAL_FEATURES)
    TEST_PREDICTED_TARGETS = CLF.predict(TEST_FEATURES)

    TRAIN_F1SCORE_MACRO = f1_score(TRAIN_INDEXED_TARGET, TRAIN_PREDICTED_TARGETS, average="macro")
    VAL_F1SCORE_MACRO = f1_score(VAL_INDEXED_TARGET, VAL_PREDICTED_TARGETS, average="macro")
    TEST_F1SCORE_MACRO = f1_score(TEST_INDEXED_TARGET, TEST_PREDICTED_TARGETS, average="macro")

    logging.debug(f"Train macro F1 score is : {TRAIN_F1SCORE_MACRO * 100:0.2f}")
    logging.debug(f"Val macro F1 score is : {VAL_F1SCORE_MACRO * 100:0.2f}")
    logging.debug(f"Test macro F1 score is : {TEST_F1SCORE_MACRO * 100:0.2f}")
