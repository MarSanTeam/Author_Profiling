from typing import List
import numpy as np
import torch
from scipy.special import softmax


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
