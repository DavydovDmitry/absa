import logging
import copy
from typing import List

from sklearn.ensemble import GradientBoostingClassifier
from gensim.models import KeyedVectors
import numpy as np

from src.target.target_classifier import TargetClassifier
from src.review.parsed_sentence import ParsedSentence
from ..score.optimal_parameter import analyse_parameter
from ..score.metrics import print_sb12


def analyze_gradient_boosting(word2vec: KeyedVectors,
                              train_sentences: List[ParsedSentence],
                              test_sentences: List[ParsedSentence],
                              seed=42):
    def _optimize_parameter(name, value):
        optimal_param = analyse_parameter(classifier_class=classifier_class,
                                          word2vec=word2vec,
                                          fixed_parameters=parameters,
                                          sentences=train_sentences,
                                          parameter_name=name,
                                          parameter_values=value)
        parameters[name] = optimal_param
        logging.info(f'{name}: {optimal_param}')

    parameters = {
        'learning_rate': 0.1,
        'n_estimators': 900,
        'min_samples_leaf': 2,
        'random_state': seed,
    }
    classifier_class = GradientBoostingClassifier

    _optimize_parameter(name='n_estimators', value=[x for x in range(100, 200, 50)])
    _optimize_parameter(name='learning_rate', value=[x for x in np.linspace(0.05, 0.5, 10)])
    _optimize_parameter(name='max_features', value=[x for x in np.linspace(0.2, 1.0, 20)])
    _optimize_parameter(name='min_samples_leaf', value=[x for x in range(1, 10)])

    # test
    sentences_pred = copy.deepcopy(test_sentences)
    classifier = classifier_class(**parameters)
    target_classifier = TargetClassifier(classifier=classifier, word2vec=word2vec)
    target_classifier.fit(sentences=train_sentences)
    sentences_pred = target_classifier.predict(sentences_pred)
    print_sb12(sentences=test_sentences, sentences_pred=sentences_pred)
