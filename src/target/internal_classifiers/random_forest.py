import logging
import copy
from typing import List

from sklearn.ensemble import RandomForestClassifier
from gensim.models import KeyedVectors
import numpy as np

from src.target.target_classifier import TargetClassifier
from src.review.parsed_sentence import ParsedSentence
from ..score.optimal_parameter_oob import analyse_parameter_oob
from ..score.metrics import print_sb12

parameters = {
    'n_estimators': 900,
    'max_features': 0.326,
    'min_samples_leaf': 2,
    'max_depth': None,
    'random_state': 42,
    'n_jobs': -1,
    'bootstrap': True,
    'oob_score': True
}


def analyze_random_forest(word2vec: KeyedVectors, train_sentences: List[ParsedSentence],
                          test_sentences: List[ParsedSentence]):
    def _optimize_parameter(name, value):
        optimal_param = analyse_parameter_oob(classifier_class=classifier_class,
                                              word2vec=word2vec,
                                              fixed_parameters=parameters,
                                              df=train_df,
                                              y=train_y,
                                              parameter_name=name,
                                              parameter_values=value)
        parameters[name] = optimal_param
        logging.info(f'{name}: {optimal_param}')

    train_df, train_y = TargetClassifier.load_data()

    #     train_df, train_y = TargetClassifier(word2vec=word2vec).get_df(
    #         sentences=train_sentences)
    #     TargetClassifier.dump_data(df=train_df, y=train_y)

    classifier_class = RandomForestClassifier

    _optimize_parameter(name='n_estimators', value=[x for x in range(100, 2000, 50)])
    _optimize_parameter(name='max_features', value=[x for x in np.linspace(0.2, 1.0, 20)])
    _optimize_parameter(name='min_samples_leaf', value=[x for x in range(1, 10)])
    _optimize_parameter(name='max_depth', value=[x for x in range(10, 50)])

    # test
    sentences_pred = copy.deepcopy(test_sentences)
    classifier = classifier_class(**parameters)
    target_classifier = TargetClassifier(classifier=classifier, word2vec=word2vec)
    target_classifier.fit(df=train_df, y=train_y)
    sentences_pred = target_classifier.predict(sentences_pred)
    print_sb12(sentences=test_sentences, sentences_pred=sentences_pred)
