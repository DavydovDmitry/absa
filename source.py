import logging
from typing import List
from functools import reduce
import datetime
import os

import numpy as np
import torch as th

from src import TEST_APPENDIX, log_path
from src.preprocess.dep_parse import load_parsed_reviews
from src import parsed_reviews_dump_path
from src.utils.embedding import get_embeddings
from src.aspect.classifier import AspectClassifier
from src.polarity.classifier import PolarityClassifier
from src.review.parsed_sentence import ParsedSentence

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


def aspect_classification(train_sentences: List[ParsedSentence],
                          test_sentences: List[ParsedSentence]):
    """Print metric for aspect classification"""
    classifier = AspectClassifier(word2vec=word2vec)
    classifier.fit(train_sentences=train_sentences)
    test_sentences_pred = classifier.predict(test_sentences)
    print(classifier.score(sentences=test_sentences, sentences_pred=test_sentences_pred))
    return test_sentences_pred


def polarity_classification(train_sentences: List[ParsedSentence],
                            test_sentences: List[ParsedSentence]):
    """Print metric for polarity classification"""
    classifier = PolarityClassifier(word2vec=word2vec)
    classifier.fit(train_sentences=train_sentences)
    test_sentences_pred = classifier.predict(test_sentences)
    print(classifier.score(sentences=test_sentences, sentences_pred=test_sentences_pred))


if __name__ == "__main__":
    np.random.seed(SEED)
    th.manual_seed(SEED)
    th.cuda.manual_seed(SEED)

    configure_logging()
    word2vec = get_embeddings()

    # preprocess_pipeline(word2vec=word2vec, is_train=True)
    # preprocess_pipeline(word2vec=word2vec, is_train=False)

    train_reviews = load_parsed_reviews(file_pathway=parsed_reviews_dump_path)
    train_sentences = [x for x in reduce(lambda x, y: x + y, train_reviews)]

    test_reviews = load_parsed_reviews(file_pathway=parsed_reviews_dump_path + TEST_APPENDIX)
    test_sentences = [x for x in reduce(lambda x, y: x + y, test_reviews)]

    aspect_classification(train_sentences=train_sentences, test_sentences=test_sentences)
    polarity_classification(train_sentences=train_sentences, test_sentences=test_sentences)
