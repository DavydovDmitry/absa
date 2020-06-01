"""
This module only for development purpose.
To run full pipeline execute run_pipeline.py
"""

import logging
from typing import List
import datetime
import os
import copy

import numpy as np
import torch as th

from absa import TEST_APPENDIX, log_path
from absa import parsed_reviews_dump_path
from absa.utils.embedding import Embeddings
from absa.utils.dump import load_dump
from absa.preprocess.pipeline import preprocess_pipeline
from absa.classification.sentence.aspect.classifier import AspectClassifier as SentenceAspectClassifier
from absa.classification.opinion.aspect.classifier import AspectClassifier as TargetAspectClassifier
from absa.classification.opinion.polarity.classifier import PolarityClassifier
from absa.review.parsed.review import ParsedReview
from absa.review.parsed.sentence import ParsedSentence

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


def sentence_aspect_classification(train_reviews: List[ParsedReview],
                                   test_reviews: List[ParsedReview]) -> List[ParsedReview]:
    # classifier = SentenceAspectClassifier.load_model()
    classifier = SentenceAspectClassifier(vocabulary=vocabulary, emb_matrix=emb_matrix)

    test_reviews_pred = copy.deepcopy(test_reviews)
    for review in test_reviews_pred:
        review.reset_targets()

    classifier.fit(train_texts=train_reviews)
    test_reviews_pred = classifier.predict(test_reviews_pred)
    logging.info(
        f'Score: {classifier.score(texts=test_reviews, texts_pred=test_reviews_pred)}')
    return test_reviews_pred


def target_aspect_classification(train_reviews: List[ParsedReview],
                                 test_reviews: List[ParsedReview],
                                 test_reviews_pred: List[ParsedReview]) -> List[ParsedReview]:
    # classifier = TargetAspectClassifier.load_model()
    classifier = TargetAspectClassifier(vocabulary=vocabulary, emb_matrix=emb_matrix)
    classifier.fit(train_texts=train_reviews, val_texts=test_reviews)

    test_reviews_pred = classifier.predict(test_reviews_pred)
    logging.info(
        f'Score: {classifier.score(texts=test_reviews, texts_pred=test_reviews_pred)}')
    return test_reviews_pred


def target_polarity_classification(
        train_sentences: List[ParsedSentence],
        test_sentences: List[ParsedSentence]) -> List[ParsedSentence]:
    # classifier = PolarityClassifier.load_model()
    classifier = PolarityClassifier(vocabulary=vocabulary, emb_matrix=emb_matrix)
    classifier.fit(train_sentences=train_sentences, val_sentences=test_sentences)

    test_sentences_pred = copy.deepcopy(test_sentences)
    for sentence in test_sentences_pred:
        sentence.reset_targets_polarities()

    test_sentences_pred = classifier.predict(test_sentences_pred)
    logging.info(
        f'Score: {classifier.score(sentences=test_sentences, sentences_pred=test_sentences_pred)}'
    )
    return test_sentences_pred


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
    target_aspect_classification(train_reviews=train_reviews,
                                 test_reviews=test_reviews,
                                 test_reviews_pred=test_reviews_pred)
    # target_polarity_classification(train_sentences=train_reviews, test_sentences=test_reviews)
