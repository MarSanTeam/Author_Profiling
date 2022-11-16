# -*- coding: utf-8 -*-
# ========================================================

"""
    Author Profiling Project:
        src:
            inferencer.py
"""

# ============================ Third Party libs ============================
import os
import logging
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer

# ============================ My packages ============================
from configuration import BaseConfig
from data_prepration import prepare_ap_data
from utils import create_sbert_user_embedding, save_output, create_user_embedding
from models.t5_personality import Classifier as PersonalityClassifier
from models.t5_irony import Classifier as IronyClassifier

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    # ---------------------- create config instance --------------------------------
    CONFIG_CLASS = BaseConfig()
    CONFIG = CONFIG_CLASS.get_config()

    CLF_PATH = "finalized_model_new.sav"

    PERSONALITY_MODEL_PATH = os.path.join(
        CONFIG.assets_dir, "personality/checkpoints/QTag-epoch=08-val_loss=0.65.ckpt")

    IRONY_MODEL_PATH = os.path.join(
        CONFIG.assets_dir, "irony/checkpoints/QTag-epoch=10-val_loss=0.45.ckpt")

    EMOTION_MODEL_PATH = os.path.join(
        CONFIG.assets_dir, "emotion/checkpoints/QTag-epoch=13-val_loss=0.45.ckpt")

    # ---------------------------------- Loading CLF ------------------------------
    CLF = pickle.load(open(CLF_PATH, "rb"))
    logging.debug("Classifier was loaded")

    # ---------------------------------- Loading data ------------------------------
    DATA = prepare_ap_data(path=CONFIG.test_data_path)
    logging.debug("Dataset was loaded")

    # ------------------------------ create LM Tokenizer instance-------------------
    TOKENIZER = T5Tokenizer.from_pretrained(CONFIG.language_model_path)
    logging.debug("T5 Tokenizer was loaded")

    # ------------------------------ create User Embeddings -------------------
    SBERT = SentenceTransformer(CONFIG.sentence_transformers_path, device=CONFIG.device)
    logging.debug("SBERT model was loaded")
    SBERT_USER_EMBEDDINGS, USER_ID = create_sbert_user_embedding(DATA, SBERT)
    logging.debug("User contextual embedding was created")

    PERSONALITY_MODEL = PersonalityClassifier.load_from_checkpoint(PERSONALITY_MODEL_PATH,
                                                                   device=CONFIG.device)
    PERSONALITY_MODEL.eval()
    logging.debug("Personality model was loaded")
    PERSONALITY_USER_EMBEDDINGS, _ = create_user_embedding(DATA, PERSONALITY_MODEL,
                                                           TOKENIZER, CONFIG.max_len)
    logging.debug("User personality embedding was created")

    IRONY_MODEL = IronyClassifier.load_from_checkpoint(IRONY_MODEL_PATH, map_location=CONFIG.device)
    IRONY_MODEL.eval()
    logging.debug("Irony model was loaded")
    IRONY_USER_EMBEDDINGS, _ = create_user_embedding(DATA, IRONY_MODEL, TOKENIZER,
                                                     CONFIG.max_len)
    logging.debug("User irony embedding was created")

    EMOTION_MODEL = IronyClassifier.load_from_checkpoint(EMOTION_MODEL_PATH,
                                                         map_location=CONFIG.device)
    EMOTION_MODEL.eval()
    logging.debug("Emotion model was loaded")
    EMOTION_USER_EMBEDDINGS, _ = create_user_embedding(DATA, EMOTION_MODEL, TOKENIZER,
                                                       CONFIG.max_len)
    logging.debug("User emotion embedding was created")

    # ------------------------------ classify the users -------------------
    SBERT_USER_EMBEDDINGS = np.squeeze(SBERT_USER_EMBEDDINGS)
    PERSONALITY_USER_EMBEDDINGS = np.squeeze(PERSONALITY_USER_EMBEDDINGS)
    IRONY_USER_EMBEDDINGS = np.squeeze(IRONY_USER_EMBEDDINGS)
    EMOTION_USER_EMBEDDINGS = np.squeeze(EMOTION_USER_EMBEDDINGS)

    FEATURES = list(np.concatenate([SBERT_USER_EMBEDDINGS,
                                    IRONY_USER_EMBEDDINGS,
                                    EMOTION_USER_EMBEDDINGS,
                                    PERSONALITY_USER_EMBEDDINGS
                                    ], axis=1))

    MODEL_OUTPUTS = CLF.predict(FEATURES)
    assert len(USER_ID) == len(MODEL_OUTPUTS)
    logging.debug("Predicted user labels")

    idx2label = {0: "NI", 1: "I"}

    for index in range(len(USER_ID)):
        save_output(path=os.path.join(CONFIG.output_path, str(USER_ID[index]) + ".xml"),
                    author_id=str(USER_ID[index]),
                    label=idx2label[MODEL_OUTPUTS[index]])
