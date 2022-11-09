# -*- coding: utf-8 -*-
# ========================================================

"""
    Author Profiling Project:
        dataset:
                dataset.py
"""

# ============================ Third Party libs ============================
from abc import ABC, abstractmethod
import pytorch_lightning as pl
import torch


# ==========================================================================


class TweetLevelDataset(ABC, torch.utils.data.Dataset):
    """
        TweetLevelDataset
    """

    def __init__(self, data: dict, tokenizer, max_len: int):
        self.texts = data["texts"]
        if "targets" in data:
            self.targets = data["targets"]
        else:
            self.targets = None
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item_index):
        """

        :param item_index:
        :return:
        """
        text = self.texts[item_index]
        batch = self.tokenizer.encode_plus(text=text,
                                           add_special_tokens=True,
                                           max_length=self.max_len,
                                           return_tensors="pt",
                                           padding="max_length",
                                           truncation=True,
                                           return_token_type_ids=True)
        if self.targets:
            target = self.targets[item_index]
            return {"input_ids": batch.input_ids.flatten(),
                    "targets": torch.tensor(target)}
        return {"input_ids": batch.input_ids.flatten()}


class DataModule(pl.LightningDataModule):
    """
        DataModule
    """

    def __init__(self, data: dict,
                 config, tokenizer):
        super().__init__()
        self.config = config
        self.data = data
        self.tokenizer = tokenizer
        self.customs_dataset = {}

    def setup(self, stage=None):
        self.customs_dataset["train_dataset"] = TweetLevelDataset(
            data=self.data["train_data"], tokenizer=self.tokenizer, max_len=self.config.max_len
        )

        self.customs_dataset["val_dataset"] = TweetLevelDataset(
            data=self.data["val_data"], tokenizer=self.tokenizer, max_len=self.config.max_len
        )

        self.customs_dataset["test_dataset"] = TweetLevelDataset(
            data=self.data["test_data"], tokenizer=self.tokenizer, max_len=self.config.max_len
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.customs_dataset["train_dataset"],
                                           batch_size=self.config.batch_size,
                                           shuffle=True, num_workers=self.config.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.customs_dataset["val_dataset"],
                                           batch_size=self.config.batch_size,
                                           num_workers=self.config.num_workers)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.customs_dataset["test_dataset"],
                                           batch_size=self.config.batch_size,
                                           num_workers=self.config.num_workers)
