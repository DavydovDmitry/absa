import logging
from typing import List
from functools import reduce
import datetime
import os
import copy

from sklearn.ensemble import RandomForestClassifier

from src import TEST_APPENDIX, log_path
from src.preprocess.dep_parse import load_parsed_reviews
from src import parsed_reviews_dump_path
from src.utils.embedding import get_embeddings
from src.target.internal_classifiers import analyze_random_forest
from src.target.target_classifier import TargetClassifier
from src.target.internal_classifiers.random_forest import parameters as random_forest_parameters
from src.target.score.metrics import print_sb12
from src.target.score.metrics import print_sb3
from src.polarity.classifier import PolarityClassifier
from src.review.parsed_sentence import ParsedSentence
from src.preprocess.pipeline import preprocess_pipeline

SEED = 42


def configure_logging():
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


def targets_extraction(train_sentences: List[ParsedSentence],
                       test_sentences: List[ParsedSentence]):
    """Print metric for target extraction"""
    classifier = RandomForestClassifier(**random_forest_parameters)
    target_classifier = TargetClassifier(classifier=classifier, word2vec=word2vec)
    target_classifier.fit(sentences=train_sentences)
    sentences_pred = target_classifier.predict(test_sentences)
    print_sb12(sentences=test_sentences, sentences_pred=sentences_pred)
    return sentences_pred

    # analyze_random_forest(word2vec=word2vec,
    #                       train_sentences=train_sentences,
    #                       test_sentences=test_sentences)


def polarity_classification(train_sentences: List[ParsedSentence],
                            test_sentences: List[ParsedSentence]):
    """Print metric for polarity classification"""
    classifier = PolarityClassifier(word2vec=word2vec)
    classifier.fit(train_sentences, test_sentences)
    test_sentences_pred = classifier.predict(test_sentences)
    print_sb3(sentences_pred=test_sentences_pred, sentences=test_sentences)


if __name__ == "__main__":
    configure_logging()
    word2vec = get_embeddings()

    preprocess_pipeline(word2vec=word2vec, is_train=True)
    preprocess_pipeline(word2vec=word2vec, is_train=False)

    train_reviews = load_parsed_reviews(file_pathway=parsed_reviews_dump_path)
    train_sentences = [x for x in reduce(lambda x, y: x + y, train_reviews)]

    test_reviews = load_parsed_reviews(file_pathway=parsed_reviews_dump_path + TEST_APPENDIX)
    test_sentences = [x for x in reduce(lambda x, y: x + y, test_reviews)]

    targets_extraction(train_sentences=train_sentences, test_sentences=test_sentences)
    polarity_classification(train_sentences=train_sentences, test_sentences=test_sentences)
