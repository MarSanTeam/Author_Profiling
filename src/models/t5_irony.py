# -*- coding: utf-8 -*-
# ========================================================

"""
    Author Profiling Project:
        models:
                t5_irony.py
"""

# ============================ Third Party libs ============================
import torch
from torch import nn
import pytorch_lightning as pl
import torchmetrics
from transformers import T5EncoderModel


# ============================ My packages ============================


class Classifier(pl.LightningModule):
    """
        Classifier
    """

    def __init__(self, num_classes, lm_path, lr, max_len, class_weights):
        super().__init__()
        self.accuracy = torchmetrics.Accuracy()
        self.f_score = torchmetrics.F1(average="none", num_classes=num_classes)
        self.f_score_total = torchmetrics.F1(average="weighted", num_classes=num_classes)
        self.max_len = max_len
        self.learning_rare = lr
        self.class_weights = class_weights

        self.model = T5EncoderModel.from_pretrained(lm_path)
        self.classifier = nn.Linear(self.model.config.d_model, num_classes)
        self.max_pool = nn.MaxPool1d(self.max_len)
        self.dense = nn.Linear(self.model.config.d_model, self.model.config.d_model)

        self.loss = nn.CrossEntropyLoss(weight=torch.FloatTensor(self.class_weights))
        self.save_hyperparameters()

    def forward(self, batch):
        inputs_ids = batch["input_ids"]
        output_encoder = self.model(inputs_ids).last_hidden_state.permute(0, 2, 1)
        maxed_pool = self.max_pool(output_encoder).squeeze(2)
        dense = self.dense(maxed_pool)
        final_output = self.classifier(dense)
        return final_output, dense

    def training_step(self, batch, batch_idx):
        """
        :param batch:
        :param batch_idx:
        :return:
        """
        label = batch["targets"].flatten()
        outputs, _ = self.forward(batch)
        loss = self.loss(outputs, label)

        metric2value = {"train_loss": loss,
                        "train_acc":
                            self.accuracy(torch.softmax(outputs, dim=1), label),
                        "train_f1_first_class":
                            self.f_score(torch.softmax(outputs, dim=1), label)[0],
                        "train_f1_second_class":
                            self.f_score(torch.softmax(outputs, dim=1), label)[1],
                        "train_f1":
                            self.f_score_total(torch.softmax(outputs, dim=1), label),
                        }

        self.log_dict(metric2value, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": label}

    def validation_step(self, batch, batch_idx):
        """
        :param batch:
        :param batch_idx:
        :return:
        """
        label = batch["targets"].flatten()
        outputs, _ = self.forward(batch)
        loss = self.loss(outputs, label)

        metric2value = {"val_loss": loss,
                        "val_acc":
                            self.accuracy(torch.softmax(outputs, dim=1), label),
                        "val_f1_first_class":
                            self.f_score(torch.softmax(outputs, dim=1), label)[0],
                        "val_f1_second_class":
                            self.f_score(torch.softmax(outputs, dim=1), label)[1],
                        "val_f1":
                            self.f_score_total(torch.softmax(outputs, dim=1), label),
                        }
        self.log_dict(metric2value, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        """
        :param batch:
        :param batch_idx:
        :return:
        """
        label = batch["targets"].flatten()
        outputs, _ = self.forward(batch)
        loss = self.loss(outputs, label)

        metric2value = {"test_loss": loss,
                        "test_acc":
                            self.accuracy(torch.softmax(outputs, dim=1), label),
                        "test_f1_first_class":
                            self.f_score(torch.softmax(outputs, dim=1), label)[0],
                        "test_f1_second_class":
                            self.f_score(torch.softmax(outputs, dim=1), label)[1],
                        "test_f1":
                            self.f_score_total(torch.softmax(outputs, dim=1), label),
                        }
        self.log_dict(metric2value, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        """
        :return:
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rare)
        return [optimizer]
