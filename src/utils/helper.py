# -*- coding: utf-8 -*-
# ========================================================

"""
    Author Profiling Project:
        utils:
                helper.py
"""

# ============================ Third Party libs ============================
from typing import List
import numpy as np
import torch
from math import sqrt
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
import xml.etree.cElementTree as ET
from torch.utils.data import DataLoader


def create_user_embedding(data, lm_model, tokenizer) -> [list, list]:
    """

    :param data:
    :param lm_model:
    :param tokenizer:
    :return:
    """
    # num_user = len(data)
    # print(f"we have {num_user} users")
    user_embeddings, user_label = [], []
    # user_counter = 0
    for author_tweets, author_label in data:
        # user_counter += 1
        author_tweets = tokenizer.batch_encode_plus(author_tweets, padding=True).input_ids
        author_tweets = torch.tensor(author_tweets)
        # print(author_tweets.size())
        # print(author_tweets)
        lm_model.to("cuda:0")
        author_tweets = author_tweets.to("cuda:0")
        with torch.no_grad():
            output = lm_model(author_tweets).pooler_output
            # print(output.size())
            output = torch.mean(output, 0)
            # print(output.size())
            user_embeddings.append(output.cpu().numpy())
            user_label.append(author_label)
        # print(f"{user_counter} user embedding created")
    return user_embeddings, user_label


def create_user_embedding_sbert(data, model) -> [list, list]:
    """

    :param data:
    :param model:
    :return:
    """
    user_embeddings, user_label = [], []
    for author_tweets, author_label in data:
        embeddings = model.encode(author_tweets)
        avg_embeddings = np.mean(embeddings, axis=0)
        user_embeddings.append(avg_embeddings)
        user_label.append(author_label)
    return user_embeddings, user_label


def create_user_embedding_irony(data: List[list], model, tokenizer) -> [list, list]:
    """

    :param data:
    :param model:
    :param tokenizer:
    :return:
    """
    user_embeddings, user_label = [], []

    for author_tweets, author_label in data:
        scores = []
        for tweet in author_tweets:
            tweet = tokenizer(tweet, return_tensors="pt")
            output = model(**tweet)
            score = torch.nn.Softmax(dim=1)(output[0])
            scores.append(score[0].detach().numpy())
        avg_embeddings = np.mean(scores, axis=0)
        user_embeddings.append(avg_embeddings)
        user_label.append(author_label)
    return user_embeddings, user_label


def create_user_embedding_personality(data: List[list], model, tokenizer, max_len) -> [list, list]:
    """

    :param data:
    :param model:
    :param tokenizer:
    :param max_len:
    :return:
    """
    user_embeddings, user_label = [], []

    for author_tweets, author_label in data:
        author_tweets = tokenizer.batch_encode_plus(author_tweets,
                                                    max_length=max_len,
                                                    padding="max_length",
                                                    truncation=True,
                                                    return_tensors="pt")
        # author_tweets = author_tweets.to("cuda:1")
        output = model(author_tweets)
        output = torch.max(output[-1], dim=0)
        output = output[0].detach().tolist()
        user_embeddings.append(output)
        user_label.append(author_label)

    return user_embeddings, user_label


def create_user_embedding_personality_1(data: List[list], model, tokenizer, max_len) -> [list, list]:
    """

    :param data:
    :param model:
    :param tokenizer:
    :param max_len:
    :return:
    """
    user_embeddings, user_label = [], []

    for author_tweets, author_label in data:
        pred = []
        dataset = InferenceDataset(data={"texts": author_tweets}, tokenizer=tokenizer, max_len=max_len)
        dataloader = DataLoader(dataset, batch_size=1,
                                shuffle=False, num_workers=4)
        for i_batch, sample_batched in enumerate(dataloader):
            sample_batched["input_ids"] = sample_batched["input_ids"].to("cuda:0")
            output = model(sample_batched)
            pred.append(output[-1].cpu().detach().numpy())
        output = np.mean(pred, axis=0)
        user_embeddings.append(output)
        user_label.append(author_label)

    return user_embeddings, user_label


def create_data_loader(texts: list, max_len: int, batch_size: int, tokenizer, dataset_obj):
    """

    :param texts:
    :param max_len:
    :param batch_size:
    :param tokenizer:
    :param dataset_obj:

    :return:
    """
    dataset = dataset_obj(data={"texts": texts},
                          max_len=max_len, tokenizer=tokenizer)

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=False, num_workers=8)
    return dataloader


def calculate_confidence_interval(metric_value: float, n_samples: int,
                                  confidence_interval_percentage: int) -> [float, float, float]:
    """

    :param metric_value:
    :param n_samples:
    :param confidence_interval_percentage:
    :return:
    """
    confidence_interval_percentage2standard_deviations_value = {
        90: 1.64,
        95: 1.96,
        98: 2.33,
        99: 2.58
    }
    confidence_interval_constant = \
        confidence_interval_percentage2standard_deviations_value[confidence_interval_percentage]
    interval = confidence_interval_constant * sqrt((metric_value * (1 - metric_value)) / n_samples)
    return interval


def get_split_data(features: list, labels: list, indexes: list) -> [list, list]:
    """

    :param features:
    :param labels:
    :param indexes:
    :return:
    """
    selected_features = [features[index] for index in indexes]
    selected_labels = [labels[index] for index in indexes]
    return selected_features, selected_labels


def cross_validator(classifier, features, labels, cv):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=1234)
    cv_scores = []
    cv_ci = []
    for train_index, test_index in skf.split(features, labels):
        x_train, y_train = get_split_data(features, labels, train_index)
        x_test, y_test = get_split_data(features, labels, test_index)
        classifier.fit(x_train, y_train)
        predictions = classifier.predict(x_test)
        acc = accuracy_score(y_test, predictions)
        ci = calculate_confidence_interval(acc, len(y_test), 95)
        cv_scores.append(acc)
        cv_ci.append(ci)
    return np.array(cv_scores), np.array(cv_ci)


def save_output(path: str, author_id: str, label: str) -> None:
    """

    :param path:
    :param author_id:
    :param label:
    :return:
    """
    author = ET.Element("author", id=str(author_id), lang="en", type=label)
    tree = ET.ElementTree(author)
    tree.write(path)
