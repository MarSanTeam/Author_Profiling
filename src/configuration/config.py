# -*- coding: utf-8 -*-
# ========================================================

"""
    Author Profiling Project:
        configuration:
                config.py
"""

# ============================ Third Party libs ============================
import argparse
from pathlib import Path
import torch


# ========================================================

class BaseConfig:
    """
    BaseConfig class is written to write configs in it
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--model_name", type=str, default="T5_Large")

        self.parser.add_argument("--raw_data_dir", type=str,
                                 default=Path(__file__).parents[2].__str__() + "/data/Raw/en")

        self.parser.add_argument("--processed_data_dir", type=str,
                                 default=Path(__file__).parents[2].__str__() + "/data/Processed")

        self.parser.add_argument("--assets_dir", type=str,
                                 default=Path(__file__).parents[2].__str__() + "/assets/")

        self.parser.add_argument("--saved_model_path", type=str,
                                 default=Path(__file__).parents[
                                             2].__str__() + "/assets/saved_models/"),

        self.parser.add_argument("--language_model_path", type=str,
                                 default="/home/LanguageModels/t5_en_large")

        self.parser.add_argument("--irony_output_file_path", type=str,
                                 default=Path(__file__).parents[2].__str__()
                                         + "/assets/irony.pkl")
        self.parser.add_argument("--personality_output_file_path", type=str,
                                 default=Path(__file__).parents[2].__str__()
                                         + "/assets/personality_mean1.pkl")
        self.parser.add_argument("--myirony_output_file_path", type=str,
                                 default=Path(__file__).parents[2].__str__()
                                         + "/assets/myirony_mean1.pkl")

        self.parser.add_argument("--emotion_output_file_path", type=str,
                                 default=Path(__file__).parents[2].__str__()
                                         + "/assets/emotion_mean1.pkl")

        self.parser.add_argument("--csv_logger_path", type=str,
                                 default=Path(__file__).parents[2].__str__() + "/assets")

        self.parser.add_argument("--truth_data", type=str, default="truth.txt")

        self.parser.add_argument("--train_data", type=str, default="train_tweet_level.csv")
        self.parser.add_argument("--val_data", type=str, default="val_tweet_level.csv")
        self.parser.add_argument("--test_data", type=str, default="test_tweet_level.csv")

        self.parser.add_argument("--test_data_path", type=str,
                                 default="../data/Raw/pan22-author-profiling-test-2022-04-22/"
                                         "pan22-author-profiling-test-2022-04-22-without_truth/en/")
        self.parser.add_argument("--output_path", type=str, default="../data/output/")

        self.parser.add_argument("--data_headers", type=list, default=["tweets", "labels"])
        self.parser.add_argument("--customized_headers", type=list, default=["texts", "targets"])

        self.parser.add_argument("--save_top_k", type=int, default=1, help="...")

        self.parser.add_argument("--num_workers", type=int,
                                 default=10,
                                 help="...")

        self.parser.add_argument("--max_len", type=int,
                                 default=2,
                                 help="...")

        self.parser.add_argument("--n_epochs", type=int,
                                 default=100,
                                 help="...")

        self.parser.add_argument("--batch_size", type=int,
                                 default=1,
                                 help="...")

        self.parser.add_argument("--lr", default=2e-5,
                                 help="...")

        self.parser.add_argument("--dropout", type=float,
                                 default=0.15,
                                 help="...")
        self.parser.add_argument("--device", type=str, default=torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"), help="")

    def get_config(self):
        """

        :return:
        """
        return self.parser.parse_args()
