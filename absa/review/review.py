"""Module for reviews representation

Also function `get_reviews` create Review instances from xml.
Assuming that xml have a structure...
"""

import xml
import re
from typing import List
import logging
import pickle
import sys

from gensim.models import KeyedVectors
from tqdm import tqdm
from .sentence import Sentence
from .target import Target
from absa import PROGRESSBAR_COLUMNS_NUM


class Review:
    def __init__(self, sentences: List[Sentence]):
        self.sentences = sentences

    def __repr__(self):
        nl = '\n'
        return f'{f"{nl}".join(map(lambda x : str(x), self.sentences))}'

    def display(self):
        """Color print of review."""

        for sentence in self.sentences:
            sentence.display()
            print(end=' ')

    def get_normalized(self) -> 'Review':
        return Review(sentences=[sentence.get_normalized() for sentence in self.sentences])


def get_reviews(root: xml.etree.ElementTree.Element, word2vec: KeyedVectors) -> List[Review]:
    def _tokenize(text: str) -> List[str]:
        """Get representation of sentence text"""
        tokens = []
        for token in re.split(pattern=r'(\w+(?:-\w+)*\s*)', string=text):
            if token:
                while re.findall(r'-\w', token):
                    temp = token.strip()

                    # check there's is no such word in vocabulary.
                    # words in vocabulary in format lemma_POS
                    # todo: word can be not in canonical form
                    if not [k for k in word2vec.vocab.keys() if temp == k[:len(temp)]]:
                        first_token, token = token.split('-', maxsplit=1)
                        tokens.append('-')
                        tokens.append(first_token)
                    else:
                        tokens.append(token)
                        break
                else:
                    tokens.append(token)
        return tokens

    logging.info('Start reviews parsing...')
    reviews = []
    with tqdm(total=len(root), ncols=PROGRESSBAR_COLUMNS_NUM, file=sys.stdout) as progress_bar:
        for review_index, review in enumerate(root):
            sentences = []
            for sentence_index, sentence in enumerate(review.find('sentences')):
                text_tokens = _tokenize(sentence.find('text').text)
                targets = []

                if sentence.find('Opinions'):
                    for opinion in sentence.find('Opinions'):
                        target = opinion.get('target')
                        category = opinion.get('category')
                        polarity = opinion.get('polarity')
                        target_slice = slice(int(opinion.get('from')), int(opinion.get('to')))

                        nodes = []
                        if target != 'NULL':
                            start = 0
                            for token_index, token in enumerate(text_tokens):
                                stop = start + len(token.strip())
                                if (target_slice.start <= start) and (stop <=
                                                                      target_slice.stop):
                                    nodes.append(token_index)
                                start += len(token)

                        targets.append(
                            Target(nodes=nodes, category=category, polarity=polarity))

                sentences.append(Sentence(text=text_tokens, targets=targets))
            reviews.append(Review(sentences))
            progress_bar.update(1)
    logging.info('Reviews parsing is complete.')
    return reviews


def dump_reviews(reviews: List[Review], file_pathway):
    with open(file_pathway, 'wb') as f:
        pickle.dump(reviews, f)
    logging.info('Make a dump of reviews.')


def load_reviews(file_pathway) -> List[Review]:
    with open(file_pathway, 'rb') as f:
        reviews = pickle.load(f)
    logging.info('Upload reviews from dump.')
    return reviews