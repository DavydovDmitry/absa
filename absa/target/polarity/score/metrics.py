"""
Set of functions to print scores for subtasks of SemEval2016 by
comparing sentences with original targets and sentences with predicted
targets.
"""

from typing import List
import logging

from absa import SCORE_DECIMAL_LEN
from absa.review.parsed_sentence import ParsedSentence


def get_sb3(sentences: List[ParsedSentence], sentences_pred: List[ParsedSentence]) -> float:
    """
        Check targets polarity.
    """
    correct_predictions = 0
    total_predictions = 0

    for s_index, (s, s_pred) in enumerate(zip(sentences, sentences_pred)):
        if len(s.targets) != len(s_pred.targets) or len(
                set(hash(t)
                    for t in s.targets) & (set(hash(t)
                                               for t in s_pred.targets))) != len(s.targets):
            print(len(set(s.targets).intersection(set(s_pred.targets))))

            logging.error('-' * 50 + ' Original  targets ' + '-' * 50)
            for l_target in s.targets:
                logging.error(l_target)
            logging.error('-' * 50 + ' Predicted targets ' + '-' * 50)
            for l_target_pred in s_pred.targets:
                logging.error(l_target_pred)
            raise ValueError(f'Targets mismatch in {s_index} sentence.')

        for target in s.targets:
            for target_pred in s_pred.targets:
                if (target.nodes == target_pred.nodes) and (target.category
                                                            == target_pred.category):
                    if target.polarity == target_pred.polarity:
                        correct_predictions += 1
                    total_predictions += 1
                    break
            else:
                raise ValueError

    accuracy = correct_predictions / total_predictions
    return accuracy


# ------------------------------------ PRINT ----------------------------------


def print_sb3(sentences: List[ParsedSentence], sentences_pred: List[ParsedSentence]):
    accuracy = get_sb3(sentences=sentences, sentences_pred=sentences_pred)
    print(f'Accuracy: {accuracy:.{SCORE_DECIMAL_LEN}f}')
