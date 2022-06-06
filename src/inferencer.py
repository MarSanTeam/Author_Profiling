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
    create_user_embedding_irony, create_user_embedding_personality, calculate_confidence_interval
from models.t5_irony import Classifier as irony_classifier
from models.t5_personality import Classifier as personality_classifier

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    # create config instance
    CONFIG_CLASS = BaseConfig()
    CONFIG = CONFIG_CLASS.get_config()

    FILENAME = "best_model_gradient_boosting.sav"
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

    PERSONALITY_TOKENIZER = T5Tokenizer.from_pretrained(CONFIG.language_model_tokenizer_path)
    logging.debug("T5 Tokenizer was loaded")

    SBERT = SentenceTransformer(CONFIG.sentence_transformers_path, device="cuda:0")
    logging.debug("SBERT model was loaded")

    IRONY_MODEL = irony_classifier.load_from_checkpoint(IRONY_MODEL_PATH)
    IRONY_MODEL.eval()
    logging.debug("Irony model was loaded")

    PERSONALITY_MODEL = personality_classifier.load_from_checkpoint(PERSONALITY_MODEL_PATH)
    PERSONALITY_MODEL.eval()
    logging.debug("Personality model was loaded")

    EMOTION_MODEL = irony_classifier.load_from_checkpoint(EMOTION_MODEL_PATH)
    EMOTION_MODEL.eval()
    logging.debug("Emotion model was loaded")

    if os.path.exists("../assets/user_embeddings_test.pkl"):
        USER_EMBEDDINGS, USER_ID = read_pickle("../assets/user_embeddings_test.pkl")
        logging.debug("User contextual embedding was loaded")

    else:
        USER_EMBEDDINGS, USER_ID = create_user_embedding_sbert(DATA, SBERT)
        write_pickle("../assets/user_embeddings_test.pkl", [USER_EMBEDDINGS, USER_ID])
        logging.debug("User contextual embedding was created")

    if os.path.exists("../assets/personality_embeddings_test.pkl"):
        USER_EMBEDDINGS_PERSONALITY = read_pickle("../assets/personality_embeddings_test.pkl")
        logging.debug("User personality embedding was loaded")
    else:
        USER_EMBEDDINGS_PERSONALITY, _ = create_user_embedding_personality(DATA,
                                                                           PERSONALITY_MODEL,
                                                                           PERSONALITY_TOKENIZER,
                                                                           CONFIG.max_len)
        logging.debug("User personality embedding was created")
        write_pickle("../assets/personality_embeddings_test.pkl", USER_EMBEDDINGS_PERSONALITY)

    if os.path.exists("../assets/irony_embeddings_test.pkl"):
        USER_EMBEDDINGS_IRONY = read_pickle("../assets/irony_embeddings_test.pkl")
        logging.debug("User irony embedding was loaded")
    else:
        USER_EMBEDDINGS_IRONY, _ = create_user_embedding_personality(DATA,
                                                                     IRONY_MODEL,
                                                                     PERSONALITY_TOKENIZER,
                                                                     CONFIG.max_len)
        logging.debug("User irony embedding was created")
        write_pickle("../assets/irony_embeddings_test.pkl", USER_EMBEDDINGS_IRONY)

    if os.path.exists("../assets/emotion_embeddings_test.pkl"):
        USER_EMBEDDINGS_EMOTION = read_pickle("../assets/emotion_embeddings_test.pkl")
        logging.debug("User emotion embedding was loaded")
    else:
        USER_EMBEDDINGS_EMOTION, _ = create_user_embedding_personality(DATA,
                                                                       EMOTION_MODEL,
                                                                       PERSONALITY_TOKENIZER,
                                                                       CONFIG.max_len)
        logging.debug("User emotion embedding was created")
        write_pickle("../assets/emotion_embeddings_test.pkl", USER_EMBEDDINGS_EMOTION)

    my_data = [USER_EMBEDDINGS, USER_EMBEDDINGS_IRONY, USER_EMBEDDINGS_EMOTION, USER_EMBEDDINGS_PERSONALITY, USER_ID]
    write_pickle("../assets/test_data.pkl", my_data)

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
