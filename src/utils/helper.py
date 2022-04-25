import numpy as np
import torch


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

