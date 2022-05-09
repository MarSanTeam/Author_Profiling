# pylint: disable-msg=too-few-public-methods
# pylint: disable-msg=no-member
# pylint: disable-msg=arguments-differ

"""
    AP Project:
        models:
            dataset
"""

# ============================ Third Party libs ============================
from abc import ABC, abstractmethod

import pytorch_lightning as pl
import torch


# ==========================================================================


class CustomDataset(ABC, torch.utils.data.Dataset):
    """
        CustomDataset is a abstract class
    """

    def __init__(self, data: dict, tokenizer, max_len: int):
        self.texts = data["texts"]
        self.targets = None
        if "targets" in data:
            self.targets = data["targets"]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    @abstractmethod
    def __getitem__(self, item_index):
        """

        :param item_index:
        :return:
        """
        texts = self.texts[item_index]
        if self.targets:
            target = self.targets[item_index]
            return texts, target
        return texts

    def single_data_tokenizer(self, text):
        batch = self.tokenizer.encode_plus(text=text,
                                           add_special_tokens=True,
                                           max_length=self.max_len,
                                           return_tensors="pt",
                                           padding="max_length",
                                           truncation=True,
                                           return_token_type_ids=True)
        return batch


class LMDataset(CustomDataset):
    """
        SeparateDataset
    """

    def __init__(self, data: dict, tokenizer, max_len: int):
        super().__init__(data, tokenizer, max_len)

    def __getitem__(self, item_index):
        texts, target = super(LMDataset, self).__getitem__(item_index)
        texts = self.single_data_tokenizer(texts)

        texts = texts.input_ids.flatten()

        return {"texts": texts,
                "targets": torch.tensor(target)}


class InferenceDataset(CustomDataset):
    """
    dataset to inference  data from model checkpoint
    """

    def __init__(self, data: dict, tokenizer, max_len):
        super(InferenceDataset, self).__init__(data, tokenizer, max_len)

    def __getitem__(self, item_index):
        text = super(InferenceDataset, self).__getitem__(item_index)

        batch = self.single_data_tokenizer(text)

        input_ids = batch.input_ids.flatten()

        return {"input_ids": input_ids}


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

    def setup(self):
        self.customs_dataset["train_dataset"] = InferenceDataset(
            data=self.data["train_data"], tokenizer=self.tokenizer, max_len=self.config.max_len
        )

        self.customs_dataset["val_dataset"] = InferenceDataset(
            data=self.data["val_data"], tokenizer=self.tokenizer, max_len=self.config.max_len
        )

        self.customs_dataset["test_dataset"] = InferenceDataset(
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
