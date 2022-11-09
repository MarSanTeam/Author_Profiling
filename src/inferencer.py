# -*- coding: utf-8 -*-
# ========================================================

"""
    Author Profiling Project:
        configuration:
                config.py
"""

# ============================ Third Party libs ============================
import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer

import logging

from configuration import BaseConfig
from data_loader import write_pickle, read_pickle
from data_prepration import prepare_ap_data
from utils import create_user_embedding_sbert, save_output, \
    create_user_embedding_irony, create_user_embedding_personality, \
    calculate_confidence_interval, create_user_embedding_personality_1
from models.t5_irony import Classifier as irony_classifier
from models.t5_personality import Classifier as personality_classifier

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    # create config instance
    CONFIG_CLASS = BaseConfig()
    CONFIG = CONFIG_CLASS.get_config()

    FILENAME = "finalized_model_new.sav"
    CLF = pickle.load(open(FILENAME, "rb"))
    logging.debug("Classifier was loaded")

    PERSONALITY_MODEL_PATH = "../assets/saved_models/personality/checkpoints/" \
                             "QTag-epoch=08-val_loss=0.65.ckpt"

    IRONY_MODEL_PATH = "../assets/saved_models/irony/checkpoints/" \
                       "QTag-epoch=10-val_loss=0.45.ckpt"

    EMOTION_MODEL_PATH = "../assets/saved_models/emotion/checkpoints/" \
                         "QTag-epoch=13-val_loss=0.45.ckpt"

    DATA = prepare_ap_data(path="../data/Raw/pan22-author-profiling-test-2022-04-22/"
                                "pan22-author-profiling-test-2022-04-22-without_truth/en/")
    logging.debug("Dataset was loaded")

    PERSONALITY_TOKENIZER = T5Tokenizer.from_pretrained(CONFIG.language_model_path)
    logging.debug("T5 Tokenizer was loaded")

    if os.path.exists("../assets/user_embeddings_test.pkl"):
        USER_EMBEDDINGS, USER_ID = read_pickle("../assets/user_embeddings_test.pkl")
        logging.debug("User contextual embedding was loaded")

    else:
        SBERT = SentenceTransformer(CONFIG.sentence_transformers_path, device="cuda:1")
        logging.debug("SBERT model was loaded")
        USER_EMBEDDINGS, USER_ID = create_user_embedding_sbert(DATA, SBERT)
        write_pickle("../assets/user_embeddings_test.pkl", [USER_EMBEDDINGS, USER_ID])
        logging.debug("User contextual embedding was created")

    if os.path.exists("../assets/personality_embeddings_test1.pkl"):
        USER_EMBEDDINGS_PERSONALITY = read_pickle("../assets/personality_embeddings_test1.pkl")
        logging.debug("User personality embedding was loaded")
    else:
        PERSONALITY_MODEL = personality_classifier.load_from_checkpoint(PERSONALITY_MODEL_PATH, map_location="cuda:1")
        PERSONALITY_MODEL.eval()
        PERSONALITY_MODEL.to("cuda:1")
        logging.debug("Personality model was loaded")
        USER_EMBEDDINGS_PERSONALITY, _ = create_user_embedding_personality_1(DATA,
                                                                             PERSONALITY_MODEL,
                                                                             PERSONALITY_TOKENIZER,
                                                                             CONFIG.max_len)
        logging.debug("User personality embedding was created")
        write_pickle("../assets/personality_embeddings_test1.pkl", USER_EMBEDDINGS_PERSONALITY)

    if os.path.exists("../assets/irony_embeddings_test1.pkl"):
        USER_EMBEDDINGS_IRONY = read_pickle("../assets/irony_embeddings_test1.pkl")
        logging.debug("User irony embedding was loaded")
    else:
        IRONY_MODEL = irony_classifier.load_from_checkpoint(IRONY_MODEL_PATH, map_location="cuda:1")
        IRONY_MODEL.eval()
        IRONY_MODEL.to("cuda:1")
        logging.debug("Irony model was loaded")
        USER_EMBEDDINGS_IRONY, _ = create_user_embedding_personality_1(DATA,
                                                                       IRONY_MODEL,
                                                                       PERSONALITY_TOKENIZER,
                                                                       CONFIG.max_len)
        logging.debug("User irony embedding was created")
        write_pickle("../assets/irony_embeddings_test1.pkl", USER_EMBEDDINGS_IRONY)

    if os.path.exists("../assets/emotion_embeddings_test1.pkl"):
        USER_EMBEDDINGS_EMOTION = read_pickle("../assets/emotion_embeddings_test1.pkl")
        logging.debug("User emotion embedding was loaded")
    else:
        EMOTION_MODEL = irony_classifier.load_from_checkpoint(EMOTION_MODEL_PATH, map_location="cuda:1")
        EMOTION_MODEL.eval()
        EMOTION_MODEL.to("cuda:1")
        logging.debug("Emotion model was loaded")
        USER_EMBEDDINGS_EMOTION, _ = create_user_embedding_personality_1(DATA,
                                                                         EMOTION_MODEL,
                                                                         PERSONALITY_TOKENIZER,
                                                                         CONFIG.max_len)
        logging.debug("User emotion embedding was created")
        write_pickle("../assets/emotion_embeddings_test1.pkl", USER_EMBEDDINGS_EMOTION)

    my_data = [USER_EMBEDDINGS, USER_EMBEDDINGS_IRONY, USER_EMBEDDINGS_EMOTION, USER_EMBEDDINGS_PERSONALITY, USER_ID]
    write_pickle("../assets/test_data.pkl", my_data)
    USER_EMBEDDINGS_IRONY = np.squeeze(USER_EMBEDDINGS_IRONY)
    USER_EMBEDDINGS_EMOTION = np.squeeze(USER_EMBEDDINGS_EMOTION)
    USER_EMBEDDINGS_PERSONALITY = np.squeeze(USER_EMBEDDINGS_PERSONALITY)

    FEATURES = list(np.concatenate([USER_EMBEDDINGS,
                                    USER_EMBEDDINGS_IRONY,
                                    USER_EMBEDDINGS_EMOTION,
                                    USER_EMBEDDINGS_PERSONALITY
                                    ], axis=1))

    MODEL_OUTPUTS = CLF.predict(FEATURES)
    logging.debug("Predicted user labels")
    print(MODEL_OUTPUTS)
    print(USER_ID)

    idx2label = {0: "NI", 1: "I"}
    output_file = "../data/output/"

    assert len(USER_ID) == len(MODEL_OUTPUTS)
    for index in range(len(USER_ID)):
        save_output(path=os.path.join(output_file, str(USER_ID[index]) + ".xml"),
                    author_id=str(USER_ID[index]),
                    label=idx2label[MODEL_OUTPUTS[index]])
