import pickle
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer

from configuration import BaseConfig
from data_prepration import prepare_ap_data
from utils import create_user_embedding_sbert, \
    create_user_embedding_irony, create_user_embedding_personality, calculate_confidence_interval
from models.t5_irony import Classifier as irony_classifier
from models.t5_personality import Classifier as personality_classifier

if __name__ == "__main__":
    # create config instance
    CONFIG_CLASS = BaseConfig()
    CONFIG = CONFIG_CLASS.get_config()

    FILENAME = "finalized_model.sav"

    PERSONALITY_MODEL_PATH = "../assets/saved_models/personality/checkpoints/" \
                             "QTag-epoch=08-val_loss=0.65.ckpt"

    IRONY_MODEL_PATH = "../assets/saved_models/irony/checkpoints/" \
                       "QTag-epoch=10-val_loss=0.45.ckpt"

    EMOTION_MODEL_PATH = "../assets/saved_models/emotion/checkpoints/" \
                         "QTag-epoch=13-val_loss=0.45.ckpt"

    LOADED_MODEL = pickle.load(open(FILENAME, "rb"))
    DATA = prepare_ap_data(path="", training_data=False)

    PERSONALITY_TOKENIZER = T5Tokenizer.from_pretrained(CONFIG.language_model_tokenizer_path)

    SBERT = SentenceTransformer(CONFIG.sentence_transformers_path, device="cuda:0")
    IRONY_MODEL = irony_classifier.load_from_checkpoint(IRONY_MODEL_PATH)
    PERSONALITY_MODEL = personality_classifier.load_from_checkpoint(PERSONALITY_MODEL_PATH)
    EMOTION_MODEL = irony_classifier.load_from_checkpoint(EMOTION_MODEL_PATH)
    IRONY_MODEL.eval()
    PERSONALITY_MODEL.eval()
    EMOTION_MODEL.eval()

    USER_EMBEDDINGS, USER_ID = create_user_embedding_sbert(DATA, SBERT)
    USER_EMBEDDINGS_PERSONALITY, _ = create_user_embedding_personality(DATA,
                                                                       PERSONALITY_MODEL,
                                                                       PERSONALITY_TOKENIZER,
                                                                       CONFIG.max_len)
    USER_EMBEDDINGS_IRONY, _ = create_user_embedding_personality(DATA,
                                                                 IRONY_MODEL,
                                                                 PERSONALITY_TOKENIZER,
                                                                 CONFIG.max_len)
    USER_EMBEDDINGS_EMOTION, _ = create_user_embedding_personality(DATA,
                                                                   EMOTION_MODEL,
                                                                   PERSONALITY_TOKENIZER,
                                                                   CONFIG.max_len)

    CLF = pickle.load(open(FILENAME, "rb"))
