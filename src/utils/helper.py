from typing import List
import numpy as np
import torch
from scipy.special import softmax
from dataset import InferenceDataset


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
        output = torch.mean(output[-1], dim=0)
        output = output.detach().tolist()
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
