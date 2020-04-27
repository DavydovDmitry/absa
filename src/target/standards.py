from functools import reduce
import os
from typing import List

import spacy_udpipe

from src.review import Review
from . import entities_path, attributes_path


def extract_aspects(reviews: List[Review]):
    extract_entities(reviews)
    extract_attributes(reviews)


def extract_entities(reviews: List[Review]):
    _extract_targets(reviews, is_entity=True)


def extract_attributes(reviews: List[Review]):
    _extract_targets(reviews, is_entity=False)


def _extract_targets(reviews: List[Review], is_entity):
    """Save normalized targets as potential standards."""

    def get_targets(aspect: str):
        """Get targets of entity or attribute."""

        def normalize_target(target):
            return ' '.join(
                map(lambda token: token.lemma_.lower() + '_' + token.pos_, udpipe(target)))

        return reduce(lambda x, y: x | y, [
            reduce(lambda x, y: x | y, [
                set([
                    normalize_target(' '.join(map(lambda node: sentence.text[node], target.nodes)))
                    for target in sentence.targets
                    if target.category.split('#')[is_entity] == aspect
                ])
                for sentence in review.sentences
            ])
            for review in reviews
        ])

    aspects_path = entities_path if is_entity else attributes_path
    is_entity = int(not is_entity)

    # Get labels
    all_categories = reduce(lambda x, y: x | y, [
        reduce(lambda x, y: x | y, [
            set([target.category for target in sentence.targets]) for sentence in review.sentences
        ]) for review in reviews
    ])
    aspects = set(category.split('#')[is_entity] for category in all_categories)

    # Extract targets
    udpipe = spacy_udpipe.load("ru")
    targets = dict()
    for aspect in aspects:
        targets[aspect] = get_targets(aspect)

    # Save inn files
    for aspect in aspects:
        with open(os.path.join(aspects_path, aspect), 'w') as f:
            f.write('\n'.join(targets[aspect]))
