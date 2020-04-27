import sys
from typing import Dict

from tqdm import tqdm
import pandas as pd
from gensim.models import KeyedVectors

from src.target.target_classifier import TargetClassifier
from .optimal_parameter import display_score


def calculate_score_oob(classifier_class: type, word2vec: KeyedVectors, fixed_parameters: Dict,
                        df: pd.DataFrame, y, parameter_name: str, parameter_val) -> float:
    classifier = classifier_class(**{**fixed_parameters, parameter_name: parameter_val})
    target_classifier = TargetClassifier(classifier=classifier, word2vec=word2vec)
    target_classifier.fit(df=df, y=y)
    return target_classifier.classifier.oob_score_


def analyse_parameter_oob(classifier_class, word2vec, fixed_parameters, df, y, parameter_name,
                          parameter_values) -> float:
    accuracies = list()
    with tqdm(total=len(parameter_values), file=sys.stdout) as progress_bar:
        for parameter_val in parameter_values:
            accuracies.append(
                calculate_score_oob(classifier_class=classifier_class,
                                    word2vec=word2vec,
                                    fixed_parameters=fixed_parameters,
                                    df=df,
                                    y=y,
                                    parameter_name=parameter_name,
                                    parameter_val=parameter_val))
            progress_bar.update(1)
    return display_score(parameter_values=parameter_values,
                         accuracies=accuracies,
                         parameter_name=parameter_name,
                         classifier_class=classifier_class,
                         score_name='Out-of-bag error score')
