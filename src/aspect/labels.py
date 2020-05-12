import numpy as np

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

# NONE_ASPECT_LABEL = 'NONE_ASPECT'
ASPECT_LABELS = [
    # NONE_ASPECT_LABEL,
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
ASPECT_LABELS = np.array(ASPECT_LABELS)
# ASPECT_LABELS = np.array(ASPECT_LABELS).reshape(-1, 1)
