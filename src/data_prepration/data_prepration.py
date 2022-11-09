# -*- coding: utf-8 -*-
# ========================================================

"""
    Author Profiling Project:
        data_preparation:
                data_preparation.py
"""

# ============================ Third Party libs ============================
from data_loader import read_xml
import os


def prepare_ap_data(path: str, author2irony: dict = None) -> list:
    """

    :param path:
    :param author2irony:
    :return:
    """
    data = []
    for file in os.listdir(path):
        if file[-4:] == ".xml":
            author_tweets = []
            author_id = file[:-4]
            root = read_xml(os.path.join(path, file))
            for child in root:
                for inch in child:
                    author_tweets.append(inch.text)
            if author2irony:
                data.append([author_tweets, author2irony[author_id]])
            else:
                data.append([author_tweets, author_id])
    return data
