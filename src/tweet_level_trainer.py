# -*- coding: utf-8 -*-
# ========================================================

"""
    Author Profiling Project:
        src:
            tweet_level_trainer.py
"""

# ============================ Third Party libs ============================
import os
import logging
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from transformers import T5Tokenizer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.utils import class_weight

# ============================ My packages ============================
from configuration import BaseConfig
from data_loader import read_csv
from indexer import Indexer
from dataset import DataModule
from models.t5_irony import Classifier

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    # ---------------------- create config instance -------------------------------
    CONFIG_CLASS = BaseConfig()
    CONFIG = CONFIG_CLASS.get_config()

    # --------------------- create CSVLogger instance ------------------------------
    LOGGER = CSVLogger(save_dir=CONFIG.saved_model_path, name=CONFIG.model_name)

    # ------------------------------ create LM Tokenizer instance-------------------
    TOKENIZER = T5Tokenizer.from_pretrained(CONFIG.language_model_path)

    # ---------------------------------- Loading datasets ---------------------------
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

    logging.debug(f"We have {len(TRAIN_DATA) + len(VAL_DATA) + len(TEST_DATA)} tweets in our data.")

    logging.debug(f"We have {len(TRAIN_DATA)} tweets in train data.")
    logging.debug(f"We have {len(VAL_DATA)} tweets in validation data.")
    logging.debug(f"We have {len(TEST_DATA)} tweets in test data.")

    # ---------------------------------- Indexer ---------------------------------
    TARGET_INDEXER = Indexer(vocabs=list(TRAIN_DATA.targets))
    TARGET_INDEXER.build_vocab2idx()
    TARGET_INDEXER.save(path=CONFIG.assets_dir)

    TRAIN_TARGETS_CONVENTIONAL = [[target] for target in list(TRAIN_DATA.targets)]
    TRAIN_INDEXED_TARGET = TARGET_INDEXER.convert_samples_to_indexes(TRAIN_TARGETS_CONVENTIONAL)

    VAL_TARGETS_CONVENTIONAL = [[target] for target in list(VAL_DATA.targets)]
    VAL_INDEXED_TARGET = TARGET_INDEXER.convert_samples_to_indexes(VAL_TARGETS_CONVENTIONAL)

    TEST_TARGETS_CONVENTIONAL = [[target] for target in list(TEST_DATA.targets)]
    TEST_INDEXED_TARGET = TARGET_INDEXER.convert_samples_to_indexes(TEST_TARGETS_CONVENTIONAL)

    # Calculate class_weights
    class_weights = class_weight.compute_class_weight(
        "balanced",
        classes=np.unique([item for sublist in TRAIN_INDEXED_TARGET for item in sublist]),
        y=np.array([item for sublist in TRAIN_INDEXED_TARGET for item in sublist]))

    # ------------------------------- Make DataLoader Dictionaries --------------------------------
    TRAIN_COLUMNS2DATA = {"texts": list(TRAIN_DATA.texts),
                          "targets": TRAIN_INDEXED_TARGET}
    VAL_COLUMNS2DATA = {"texts": list(VAL_DATA.texts),
                        "targets": VAL_INDEXED_TARGET}
    TEST_COLUMNS2DATA = {"texts": list(TEST_DATA.texts),
                         "targets": TRAIN_INDEXED_TARGET}

    DATA = {"train_data": TRAIN_COLUMNS2DATA,
            "val_data": VAL_COLUMNS2DATA, "test_data": TEST_COLUMNS2DATA}

    # ----------------------------- Create Data Module ------------------------------
    DATA_MODULE = DataModule(data=DATA, config=CONFIG, tokenizer=TOKENIZER)
    DATA_MODULE.setup()
    # ----------------------------- Create Trainer ----------------------------------
    CHECKPOINT_CALLBACK = ModelCheckpoint(monitor="val_loss",
                                          filename="QTag-{epoch:02d}-{val_loss:.2f}",
                                          save_top_k=CONFIG.save_top_k,
                                          mode="min")

    EARLY_STOPPING_CALLBACK = EarlyStopping(monitor="val_loss", patience=5)

    # Instantiate the Model Trainer
    TRAINER = pl.Trainer(max_epochs=CONFIG.n_epochs, gpus=[1],
                         callbacks=[CHECKPOINT_CALLBACK, EARLY_STOPPING_CALLBACK],
                         progress_bar_refresh_rate=60, logger=LOGGER)

    # --------------------------- Create classifier object ---------------------------
    MODEL = Classifier(num_classes=len(TARGET_INDEXER.get_vocab2idx()),
                       lm_path=CONFIG.language_model_path, lr=CONFIG.lr, max_len=CONFIG.max_len,
                       class_weights=class_weights)

    # ---------------------------- Train and Test Model -------------------------------
    TRAINER.fit(MODEL, datamodule=DATA_MODULE)
    TRAINER.test(ckpt_path="best", datamodule=DATA_MODULE)
