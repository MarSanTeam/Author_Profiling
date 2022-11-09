# -*- coding: utf-8 -*-
# ========================================================

"""
    Author Profiling Project:
        data_preparation:
                helper.py
"""


# ============================ Third Party libs ============================

def create_author_label(data: list) -> dict:
    """

    :param data:
    :return:
    """
    author2label = {}
    for sample in data:
        sample = sample.strip().split(":::")
        author2label[sample[0]] = sample[1]

    return author2label
