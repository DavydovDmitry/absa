"""
Set of functions to print scores for subtasks of SemEval2016 by
comparing sentences with original targets and sentences with predicted
targets.
"""

from typing import List
from collections import namedtuple
import logging

from src import SCORE_DECIMAL_LEN
from src.review.parsed_sentence import ParsedSentence

Score = namedtuple('Score', ['precision', 'recall', 'f1'])


def get_sb12(sentences: List[ParsedSentence], sentences_pred: List[ParsedSentence]) -> Score:
    """
        Check target nodes (words) and it's aspect category.
    """
    total_targets = 0
    total_predictions = 0
    correct_predictions = 0

    for sentence_index in range(len(sentences)):
        for y in sentences[sentence_index].targets:
            for y_pred in sentences_pred[sentence_index].targets:
                if (y.nodes == y_pred.nodes) and (y.category == y_pred.category):
                    correct_predictions += 1
                    break
        total_targets += len(sentences[sentence_index].targets)
        total_predictions += len(sentences_pred[sentence_index].targets)

    precision = correct_predictions / total_targets
    recall = correct_predictions / total_predictions
    f1 = 2 * (precision * recall) / (precision + recall)

    return Score(precision=precision, recall=recall, f1=f1)


def get_sb1(sentences: List[ParsedSentence], sentences_pred: List[ParsedSentence]) -> Score:
    """
        Check aspect categories for sentence.
    """
    total_targets = 0
    total_predictions = 0
    correct_predictions = 0

    for sentence_index in range(len(sentences)):
        y = set([x.category for x in sentences[sentence_index].targets])
        y_pred = set([x.category for x in sentences_pred[sentence_index].targets])
        correct_predictions += len(y & y_pred)
        total_targets += len(y)
        total_predictions += len(y_pred)

    precision = correct_predictions / total_targets
    recall = correct_predictions / total_predictions
    f1 = 2 * (precision * recall) / (precision + recall)

    return Score(precision=precision, recall=recall, f1=f1)


def get_sb2(sentences: List[ParsedSentence], sentences_pred: List[ParsedSentence]) -> Score:
    """
        Check target nodes (words).
    """
    total_targets = 0
    total_predictions = 0
    correct_predictions = 0

    for sentence_index in range(len(sentences)):
        targets = set([y.nodes for y in sentences[sentence_index].targets if y.nodes])
        pred_targets = set([
            set(y_pred.nodes) for y_pred in sentences_pred[sentence_index].targets
            if y_pred.nodes
        ])

        correct_predictions += len(targets & pred_targets)
        total_targets += len(targets)
        total_predictions += len(pred_targets)

    precision = correct_predictions / total_targets
    recall = correct_predictions / total_predictions
    f1 = 2 * (precision * recall) / (precision + recall)

    return Score(precision=precision, recall=recall, f1=f1)


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
                if (target.nodes == target_pred.nodes) and (
                        target.category == target_pred.category):
                    if target.polarity == target_pred.polarity:
                        correct_predictions += 1
                    total_predictions += 1
                    break
            else:
                raise ValueError

    accuracy = correct_predictions / total_predictions
    return accuracy


# ------------------------------------ PRINT ----------------------------------


def _print_f1_score(precision, recall, f1, decimal_len=SCORE_DECIMAL_LEN):
    logging.info(f'Precision: {precision:.{decimal_len}f}')
    logging.info(f'Recall: {recall:.{decimal_len}f}')
    logging.info(f'F1: {f1:.{decimal_len}f}')


def print_sb12(sentences: List[ParsedSentence], sentences_pred: List[ParsedSentence]):
    precision, recall, f1 = get_sb12(sentences=sentences, sentences_pred=sentences_pred)
    _print_f1_score(precision=precision, recall=recall, f1=f1)


def print_sb1(sentences: List[ParsedSentence], sentences_pred: List[ParsedSentence]):

    precision, recall, f1 = get_sb1(sentences=sentences, sentences_pred=sentences_pred)
    _print_f1_score(precision, recall, f1)


def print_sb2(sentences: List[ParsedSentence], sentences_pred: List[ParsedSentence]):
    precision, recall, f1 = get_sb2(sentences=sentences, sentences_pred=sentences_pred)
    _print_f1_score(precision, recall, f1)


def print_sb3(sentences: List[ParsedSentence], sentences_pred: List[ParsedSentence]):
    accuracy = get_sb3(sentences=sentences, sentences_pred=sentences_pred)
    print(f'Accuracy: {accuracy:.{SCORE_DECIMAL_LEN}f}')
