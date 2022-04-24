# pylint: disable-msg=no-member
# pylint: disable=too-many-ancestors
# pylint: disable=arguments-differ
# pylint: disable=unused-argument
"""
    FAQ Project:
        models:
            mt5 encoder finetune
"""
# ============================ Third Party libs ============================
import torch
from torch import nn
import pytorch_lightning as pl
import torchmetrics

# ============================ My packages ============================
from transformers import BertModel, AutoModel


class Classifier(pl.LightningModule):
    """
        Classifier
    """

    def __init__(self, num_classes, lm_path, lr, max_len):
        super().__init__()
        self.accuracy = torchmetrics.Accuracy()
        self.f_score = torchmetrics.F1(average='none', num_classes=num_classes)
        self.f_score_total = torchmetrics.F1(average="weighted", num_classes=num_classes)
        self.max_len = max_len
        self.learning_rare = lr

        self.model = BertModel.from_pretrained(lm_path, return_dict=True)
        self.classifier = nn.Linear(self.model.config.hidden_size, num_classes)

        self.loss = nn.CrossEntropyLoss()
        self.save_hyperparameters()

    def forward(self, batch):
        inputs_ids = batch["texts"]
        output_encoder = self.model(inputs_ids).pooler_output
        final_output = self.classifier(output_encoder)
        return final_output

    def training_step(self, batch, batch_idx):
        """

        :param batch:
        :param batch_idx:
        :return:
        """
        label = batch["targets"].flatten()
        outputs = self.forward(batch)
        loss = self.loss(outputs, label)

        metric2value = {"train_loss": loss,
                        "train_acc":
                            self.accuracy(torch.softmax(outputs, dim=1), label),
                        "train_f_score":
                            self.f_score(torch.softmax(outputs, dim=1), label)}

        self.log_dict(metric2value, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": label}

    def validation_step(self, batch, batch_idx):
        """

        :param batch:
        :param batch_idx:
        :return:
        """
        label = batch["targets"].flatten()
        outputs = self.forward(batch)
        loss = self.loss(outputs, label)

        metric2value = {"val_loss": loss,
                        "val_acc":
                            self.accuracy(torch.softmax(outputs, dim=1), label),
                        "val_f1_score":
                            self.f_score(torch.softmax(outputs, dim=1), label)}

        self.log_dict(metric2value, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        """

        :param batch:
        :param batch_idx:
        :return:
        """
        label = batch["targets"].flatten()
        outputs = self.forward(batch)
        loss = self.loss(outputs, label)

        metric2value = {"test_loss": loss,
                        "test_acc":
                            self.accuracy(torch.softmax(outputs, dim=1), label),
                        "test_f1_score":
                            self.f_score(torch.softmax(outputs, dim=1), label)}

        self.log_dict(metric2value, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        """

        :return:
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rare)
        return [optimizer]
