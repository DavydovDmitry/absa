from typing import List

import numpy as np


class Labels:
    def __init__(self, labels: List, none_value=None):
        self.none_value = none_value
        if none_value is not None:
            self.labels = np.array([none_value] + labels)
        else:
            self.labels = np.array(labels)

    def get_index(self, label: str):
        """
        row index     column index
         |   ___________|
         |  |
         v  v
        [0][0]
        """
        return np.where(self.labels == label)[0][0]

    def __getitem__(self, item):
        return self.labels[item]

    def __len__(self):
        return self.labels.shape[0]


POS_LABELS = [
    'ADJ',
    'ADP',
    'ADV',
    'AUX',
    'CCONJ',
    'DET',
    'INTJ',
    'NOUN',
    'NUM',
    'PART',
    'PRON',
    'PROPN',
    'PUNCT',
    'SCONJ',
    'SYM',
    'VERB',
    'X',
]
POS_LABELS = np.array(POS_LABELS).reshape(-1, 1)

DEP_LABELS = [
    'amod', 'root', 'nmod', 'obj', 'nsubj', 'case', 'det', 'obl', 'cc', 'conj', 'advmod',
    'iobj', 'xcomp', 'cop', 'nummod', 'mark', 'fixed', 'advcl', 'aux', 'acl', 'orphan',
    'csubj', 'parataxis', 'ccomp', 'appos', 'discourse', 'flat', 'punct', 'compound', 'expl'
]
DEP_LABELS = np.array(DEP_LABELS).reshape(-1, 1)

ASPECT_LABELS = [
    'SERVICE#GENERAL',
    'AMBIENCE#GENERAL',
    'FOOD#QUALITY',
    'FOOD#PRICES',
    'FOOD#STYLE_OPTIONS',
    'RESTAURANT#GENERAL',
    'RESTAURANT#PRICES',
    'DRINKS#STYLE_OPTIONS',
    'RESTAURANT#MISCELLANEOUS',
    'DRINKS#QUALITY',
    'LOCATION#GENERAL',
    'DRINKS#PRICES',
]
# ASPECT_LABELS = np.array(ASPECT_LABELS)
# ASPECT_LABELS = np.array(ASPECT_LABELS).reshape(-1, 1)
