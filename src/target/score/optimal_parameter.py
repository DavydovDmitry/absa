import sys
from typing import Dict, List

from tqdm import tqdm
from gensim.models import KeyedVectors
from sklearn.model_selection import KFold

from src.target.miner import TargetMiner
from src.review.parsed_sentence import ParsedSentence
from .display import display_score

N_SPLITS = 5


def calculate_score(sentences: List[ParsedSentence], classifier_class: type,
                    word2vec: KeyedVectors, fixed_parameters: Dict, parameter_name: str,
                    parameter_val) -> float:
    """
    Return mean score
    """
    scores = list()
    for train_indexes, test_indexes in KFold(n_splits=N_SPLITS).split(sentences):
        classifier = classifier_class(**{**fixed_parameters, parameter_name: parameter_val})
        target_classifier = TargetMiner(classifier=classifier, word2vec=word2vec)
        target_classifier.fit(sentences=[x for x in map(sentences.__getitem__, train_indexes)])
        scores.append(
            target_classifier.score(
                sentences=[x for x in map(sentences.__getitem__, test_indexes)]).f1)
    return sum(scores) / len(scores)


def analyse_parameter(sentences: List[ParsedSentence], classifier_class: type,
                      word2vec: KeyedVectors, fixed_parameters: Dict, parameter_name: str,
                      parameter_values: List) -> float:
    accuracies = list()
    with tqdm(total=len(parameter_values), file=sys.stdout) as progress_bar:
        for parameter_val in parameter_values:
            accuracies.append(
                calculate_score(sentences=sentences,
                                classifier_class=classifier_class,
                                word2vec=word2vec,
                                fixed_parameters=fixed_parameters,
                                parameter_name=parameter_name,
                                parameter_val=parameter_val))
            progress_bar.update(1)
    return display_score(parameter_values=parameter_values,
                         accuracies=accuracies,
                         parameter_name=parameter_name,
                         classifier_class=classifier_class)
