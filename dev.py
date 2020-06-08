"""
This module only for development purpose.
To run full pipeline execute run_pipeline.py
"""

import logging
from typing import List, Dict
import datetime
import os
import copy

import numpy as np
import torch as th

from absa import TEST_APPENDIX, train_reviews_path, test_reviews_path, \
    parsed_reviews_dump_path, checked_reviews_dump_path, raw_reviews_dump_path, log_path
from absa.text.parsed.review import ParsedReview
from absa.utils.nlp import NLPPipeline
from absa.utils.dump import make_dump, load_dump
from absa.utils.embedding import Embeddings
from absa.input.semeval2016 import from_xml
from absa.preprocess.spell_check import spell_check
from absa.preprocess.dependency import dep_parse_reviews
from absa.classification.sentence.aspect.classifier import AspectClassifier as SentenceAspectClassifier
from absa.classification.opinion.aspect.classifier import AspectClassifier as OpinionAspectClassifier
from absa.classification.opinion.polarity.classifier import PolarityClassifier

SEED = 42


def configure_logging():
    """Logging configuration

    Log formatting.
    Pass logs to terminal and to file.
    """
    if not os.path.isdir(log_path):
        os.mkdir(log_path)

    logging.basicConfig(level=logging.INFO)
    log_formatter = logging.Formatter(
        "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    log_formatter.formatTime = lambda record, datefmt: datetime.datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S")
    root_logger = logging.getLogger()

    file_handler = logging.FileHandler(
        f'{log_path}/{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.log')
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)


def preprocess_pipeline(vocabulary: Dict = None,
                        is_train: bool = True,
                        skip_spell_check: bool = True,
                        make_dumps: bool = True) -> List[ParsedReview]:

    # Set postfix for train/test data
    if is_train:
        appendix = ''
        reviews_path = train_reviews_path
    else:
        appendix = TEST_APPENDIX
        reviews_path = test_reviews_path

    # Parse reviews
    if os.path.isfile(raw_reviews_dump_path + appendix):
        reviews = load_dump(pathway=raw_reviews_dump_path + appendix)
    else:
        if vocabulary is None:
            raise ValueError(
                "There's no dump. Need to parse review. Therefore vocabulary parameter is required."
            )
        reviews = from_xml(pathway=reviews_path)
        if make_dumps:
            make_dump(obj=reviews, pathway=raw_reviews_dump_path + appendix)

    # Spellcheck
    if not skip_spell_check:
        if os.path.isfile(checked_reviews_dump_path + appendix):
            reviews, spell_checked2init = load_dump(pathway=checked_reviews_dump_path +
                                                    appendix)
        else:
            reviews = spell_check(reviews)
            if make_dumps:
                make_dump(obj=reviews, pathway=checked_reviews_dump_path + appendix)

    # Dependency parsing
    if os.path.isfile(parsed_reviews_dump_path + appendix):
        parsed_reviews = load_dump(pathway=parsed_reviews_dump_path + appendix)
    else:
        nlp = NLPPipeline.nlp
        parsed_reviews = dep_parse_reviews(reviews, nlp)
        if make_dumps:
            make_dump(obj=parsed_reviews, pathway=parsed_reviews_dump_path + appendix)

    return parsed_reviews


def sentence_aspect_classification(train_reviews: List[ParsedReview],
                                   test_reviews: List[ParsedReview]) -> List[ParsedReview]:
    if True:
        classifier = SentenceAspectClassifier.load_model()
    else:
        classifier = SentenceAspectClassifier(vocabulary=vocabulary, emb_matrix=emb_matrix)
        classifier.fit(train_texts=train_reviews)

    test_reviews_pred = copy.deepcopy(test_reviews)
    for review in test_reviews_pred:
        review.reset_opinions()

    test_reviews_pred = classifier.predict(test_reviews_pred)
    logging.info(
        f'Score: {classifier.score(texts=test_reviews, texts_pred=test_reviews_pred)}')
    return test_reviews_pred


def opinion_aspect_classification(train_reviews: List[ParsedReview],
                                  test_reviews: List[ParsedReview],
                                  test_reviews_pred: List[ParsedReview]) -> List[ParsedReview]:
    if True:
        classifier = OpinionAspectClassifier.load_model()
    else:
        classifier = OpinionAspectClassifier(vocabulary=vocabulary, emb_matrix=emb_matrix)
        classifier.fit(train_texts=train_reviews, val_texts=test_reviews)

    test_reviews_pred = classifier.predict(test_reviews_pred)
    logging.info(
        f'Score: {classifier.score(texts=test_reviews, texts_pred=test_reviews_pred)}')
    return test_reviews_pred


def opinion_polarity_classification(train_reviews: List[ParsedReview],
                                    test_reviews: List[ParsedReview]) -> List[ParsedReview]:
    if True:
        classifier = PolarityClassifier.load_model()
    else:
        classifier = PolarityClassifier(vocabulary=vocabulary, emb_matrix=emb_matrix)
        classifier.fit(train_texts=train_reviews, val_texts=test_reviews, num_epoch=20)

    test_reviews_pred = copy.deepcopy(test_reviews)
    for review in test_reviews_pred:
        for sentence in review.sentences:
            sentence.reset_opinions_polarities()

    test_reviews_pred = classifier.predict(test_reviews_pred)
    logging.info(
        f'Score: {classifier.score(texts=test_reviews, texts_pred=test_reviews_pred)}')
    return test_reviews_pred


if __name__ == "__main__":
    np.random.seed(SEED)
    th.manual_seed(SEED)
    th.cuda.manual_seed(SEED)

    configure_logging()
    vocabulary = Embeddings.vocabulary
    emb_matrix = Embeddings.embeddings_matrix

    preprocess_pipeline(vocabulary=vocabulary, is_train=True, skip_spell_check=False)
    preprocess_pipeline(vocabulary=vocabulary, is_train=False, skip_spell_check=False)

    train_reviews = load_dump(pathway=parsed_reviews_dump_path)
    test_reviews = load_dump(pathway=parsed_reviews_dump_path + TEST_APPENDIX)

    test_reviews_pred = sentence_aspect_classification(train_reviews=train_reviews,
                                                       test_reviews=test_reviews)
    opinion_aspect_classification(train_reviews=train_reviews,
                                  test_reviews=test_reviews,
                                  test_reviews_pred=test_reviews_pred)
    opinion_polarity_classification(train_reviews=train_reviews, test_reviews=test_reviews)
