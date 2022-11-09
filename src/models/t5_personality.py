# -*- coding: utf-8 -*-
# ========================================================

"""
    Author Profiling Project:
        models:
                t5_personality.py
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
        self.dense = nn.Linear(self.model.config.d_model, self.model.config.d_model)

        self.classifier_1 = nn.Linear(self.model.config.d_model, num_classes)
        self.classifier_2 = nn.Linear(self.model.config.d_model, num_classes)
        self.classifier_3 = nn.Linear(self.model.config.d_model, num_classes)
        self.classifier_4 = nn.Linear(self.model.config.d_model, num_classes)

        self.max_pool = nn.MaxPool1d(self.max_len)

        self.loss_1 = nn.CrossEntropyLoss(weight=torch.FloatTensor(self.class_weights[0]))
        self.loss_2 = nn.CrossEntropyLoss(weight=torch.FloatTensor(self.class_weights[1]))
        self.loss_3 = nn.CrossEntropyLoss(weight=torch.FloatTensor(self.class_weights[2]))
        self.loss_4 = nn.CrossEntropyLoss(weight=torch.FloatTensor(self.class_weights[3]))
        self.save_hyperparameters()

    def forward(self, batch):
        inputs_ids = batch["input_ids"]
        output_encoder = self.model(inputs_ids).last_hidden_state.permute(0, 2, 1)
        maxed_pool = self.max_pool(output_encoder).squeeze(2)

        dense = self.dense(maxed_pool)

        final_output_1 = self.classifier_1(dense)
        final_output_2 = self.classifier_2(dense)
        final_output_3 = self.classifier_3(dense)
        final_output_4 = self.classifier_4(dense)
        return final_output_1, final_output_2, final_output_3, final_output_4, maxed_pool

    def training_step(self, batch, batch_idx):
        """
        :param batch:
        :param batch_idx:
        :return:
        """
        label_1 = batch["targets_1"].flatten()
        label_2 = batch["targets_2"].flatten()
        label_3 = batch["targets_3"].flatten()
        label_4 = batch["targets_4"].flatten()
        outputs_1, outputs_2, outputs_3, outputs_4, _ = self.forward(batch)
        loss_1 = self.loss_1(outputs_1, label_1)
        loss_2 = self.loss_2(outputs_1, label_2)
        loss_3 = self.loss_3(outputs_1, label_3)
        loss_4 = self.loss_4(outputs_1, label_4)
        loss = (loss_1 + loss_2 + loss_3 + loss_4) / 4

        metric2value = {"train_loss": loss,
                        "train_f1_1":
                            self.f_score_total(torch.softmax(outputs_1, dim=1), label_1),
                        "train_f1_2":
                            self.f_score_total(torch.softmax(outputs_2, dim=1), label_2),
                        "train_f1_3":
                            self.f_score_total(torch.softmax(outputs_3, dim=1), label_3),
                        "train_f1_4":
                            self.f_score_total(torch.softmax(outputs_4, dim=1), label_4),
                        }

        self.log_dict(metric2value, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs_1, "labels": label_1}

    def validation_step(self, batch, batch_idx):
        """
        :param batch:
        :param batch_idx:
        :return:
        """
        label_1 = batch["targets_1"].flatten()
        label_2 = batch["targets_2"].flatten()
        label_3 = batch["targets_3"].flatten()
        label_4 = batch["targets_4"].flatten()
        outputs_1, outputs_2, outputs_3, outputs_4, _ = self.forward(batch)
        loss_1 = self.loss_1(outputs_1, label_1)
        loss_2 = self.loss_2(outputs_1, label_2)
        loss_3 = self.loss_3(outputs_1, label_3)
        loss_4 = self.loss_4(outputs_1, label_4)
        loss = (loss_1 + loss_2 + loss_3 + loss_4) / 4

        metric2value = {"val_loss": loss,
                        "val_f1_1":
                            self.f_score_total(torch.softmax(outputs_1, dim=1), label_1),
                        "val_f1_2":
                            self.f_score_total(torch.softmax(outputs_2, dim=1), label_2),
                        "val_f1_3":
                            self.f_score_total(torch.softmax(outputs_3, dim=1), label_3),
                        "val_f1_4":
                            self.f_score_total(torch.softmax(outputs_4, dim=1), label_4),
                        }

        self.log_dict(metric2value, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        """
        :param batch:
        :param batch_idx:
        :return:
        """
        label_1 = batch["targets_1"].flatten()
        label_2 = batch["targets_2"].flatten()
        label_3 = batch["targets_3"].flatten()
        label_4 = batch["targets_4"].flatten()
        outputs_1, outputs_2, outputs_3, outputs_4, _ = self.forward(batch)
        loss_1 = self.loss_1(outputs_1, label_1)
        loss_2 = self.loss_2(outputs_1, label_2)
        loss_3 = self.loss_3(outputs_1, label_3)
        loss_4 = self.loss_4(outputs_1, label_4)
        loss = (loss_1 + loss_2 + loss_3 + loss_4) / 4

        metric2value = {"test_loss": loss,
                        "test_f1_1":
                            self.f_score_total(torch.softmax(outputs_1, dim=1), label_1),
                        "test_f1_2":
                            self.f_score_total(torch.softmax(outputs_2, dim=1), label_2),
                        "test_f1_3":
                            self.f_score_total(torch.softmax(outputs_3, dim=1), label_3),
                        "test_f1_4":
                            self.f_score_total(torch.softmax(outputs_4, dim=1), label_4),
                        }

        self.log_dict(metric2value, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        """
        :return:
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rare)
        return [optimizer]
