from typing import List, Dict, Union, Tuple
import logging
import time
import copy
import sys
from dataclasses import dataclass

import torch as th
import numpy as np
from gensim.models import KeyedVectors
from frozendict import frozendict
from scipy.optimize import minimize as minimize
from tqdm import tqdm

from absa import sentence_aspect_classifier_dump_path, SCORE_DECIMAL_LEN, PROGRESSBAR_COLUMNS_NUM
from absa.review.parsed_sentence import ParsedSentence
from absa.review.target import Target
from .loader import DataLoader
from .nn.nn import NeuralNetwork
from absa.labels.labels import Labels
from absa.labels.default import ASPECT_LABELS


class AspectClassifier:
    """Sentence level aspect classifier

    Attributes
    ----------
    word2vec : KeyedVectors
        Vocabulary and embed_matrix are extracting from word2vec.
        Otherwise you can pass vocabulary and emb_matrix.
    vocabulary : dict
        dictionary where key is wordlemma_POS value - index in embedding matrix
    emb_matrix : th.Tensor
        matrix of pretrained embeddings
    """
    def __init__(self,
                 word2vec: KeyedVectors = None,
                 vocabulary: Dict[str, int] = None,
                 emb_matrix: th.Tensor = None,
                 aspect_labels: List[str] = ASPECT_LABELS,
                 batch_size: int = 100):
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.aspect_labels = Labels(labels=aspect_labels)

        # prepare vocabulary and embeddings
        if word2vec is not None:
            self.vocabulary = {w: i for i, w in enumerate(word2vec.index2word)}
            emb_matrix = th.FloatTensor(word2vec.vectors)
        else:
            self.vocabulary = vocabulary

        while True:
            try:
                self.model = NeuralNetwork(emb_matrix=emb_matrix,
                                           device=self.device,
                                           num_class=len(self.aspect_labels)).to(self.device)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    th.cuda.empty_cache()
            else:
                break

        criterion = th.nn.BCEWithLogitsLoss()
        self.loss_func = lambda y_pred, y: criterion(y_pred, y)

    def fit(self,
            train_sentences: List[ParsedSentence],
            val_sentences: List[ParsedSentence] = None,
            optimizer_class=th.optim.Adam,
            optimizer_params: Dict = frozendict({
                'lr': 0.01,
            }),
            num_epoch=50,
            init_threshold: np.array = None,
            fixed_threshold=False,
            save_state=True,
            verbose=False) -> Union[np.array, Tuple[np.array, np.array]]:
        """Fit on train sentences and save model state.

        Every epoch consist from 2 stages
        - Train stage
            Optimize parameters of neural network and select optimal
            threshold for every class.
        - Validation
            Calculate scores on unseen sentences.
        """

        if init_threshold is None:
            self.threshold = np.random.random(len(self.aspect_labels)) - 1.0
        else:
            self.threshold = init_threshold

        parameters = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = optimizer_class(parameters, **optimizer_params)

        train_batches = DataLoader(sentences=train_sentences,
                                   batch_size=self.batch_size,
                                   vocabulary=self.vocabulary,
                                   aspect_labels=self.aspect_labels,
                                   device=self.device)
        train_f1_history = np.empty(shape=(num_epoch, ), dtype=np.float)

        if val_sentences:
            val_batches = DataLoader(sentences=val_sentences,
                                     batch_size=self.batch_size,
                                     vocabulary=self.vocabulary,
                                     aspect_labels=self.aspect_labels,
                                     device=self.device)
            val_f1_history = np.empty(shape=(num_epoch, ), dtype=np.float)

        def epoch_step():
            # Train
            self.model.train()
            train_logits, train_labels = [], []
            for i, batch in enumerate(train_batches):
                optimizer.zero_grad()
                logits = self.model(embed_ids=batch.embed_ids, sentence_len=batch.sentence_len)
                loss = self.loss_func(logits, batch.labels)
                loss.backward()
                optimizer.step()

                train_logits.append(logits.to('cpu').data.numpy())
                train_labels.append(batch.labels.to('cpu').data.numpy())
            train_logits = np.concatenate(train_logits)
            train_labels = np.concatenate(train_labels)
            if not fixed_threshold:
                opt_results = minimize(lambda threshold: -self.f1_score(
                    logits=train_logits, labels=train_labels, threshold=threshold),
                                       x0=self.threshold)
                self.threshold = opt_results.x
                f1_score = -opt_results.fun
            else:
                f1_score = self.f1_score(logits=train_logits,
                                         labels=train_labels,
                                         threshold=self.threshold)
            train_f1_history[epoch] = f1_score

            # Validation
            if val_sentences:
                val_logits, val_labels = [], []
                self.model.eval()
                for i, batch in enumerate(val_batches):
                    logit = self.model(embed_ids=batch.embed_ids,
                                       sentence_len=batch.sentence_len)
                    val_logits.append(logit.to('cpu').data.numpy())
                    val_labels.append(batch.labels.to('cpu').data.numpy())
                val_logits = np.concatenate(val_logits)
                val_labels = np.concatenate(val_labels)
                val_f1_history[epoch] = self.f1_score(logits=val_logits,
                                                      labels=val_labels,
                                                      threshold=self.threshold)

            # todo: logging

        if verbose:
            for epoch in range(num_epoch):
                epoch_step()
        else:
            with tqdm(total=num_epoch, ncols=PROGRESSBAR_COLUMNS_NUM,
                      file=sys.stdout) as progress_bar:
                for epoch in range(num_epoch):
                    epoch_step()
                    progress_bar.update(1)

        if save_state:
            self.save_model()
        if val_sentences is not None:
            return train_f1_history, val_f1_history
        return train_f1_history

    @staticmethod
    def f1_score(logits: np.array, labels: np.array, threshold: np.array) -> float:
        """Calculate f1 score from nn output.

        Parameters
        ----------
        logits : np.array
            neural network predictions for every aspect.
        labels : np.array
            array of {0, 1} where:
        1 - aspect present in sentence, 0 - otherwise.
        threshold : np.array
            threshold for aspect selection.

        Returns
        -------
        f1_score : float
            macro f1 score
        """
        labels_pred = np.where(logits > threshold, 1, 0)
        correct = labels[labels_pred.nonzero()].sum()
        total_predictions = labels_pred.sum()
        total_labels = labels.sum()

        precision = correct / total_predictions
        recall = correct / total_labels
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

    def predict(self, sentences: List[ParsedSentence]) -> List[ParsedSentence]:
        """Modify passed sentences. Add targets with empty list of nodes.

        Parameters
        ----------
        sentences : List[ParsedSentence]
            Sentences with extracted targets.

        Return
        -------
        sentences : List[ParsedSentence]
            Sentences with defined targets.
        """
        self.model.eval()
        sentences = copy.deepcopy(sentences)
        for sentence in sentences:
            sentence.reset_targets()

        batches = DataLoader(sentences=sentences,
                             batch_size=self.batch_size,
                             vocabulary=self.vocabulary,
                             aspect_labels=self.aspect_labels,
                             device=self.device)
        for batch_index, batch in enumerate(batches):
            logits = self.model(embed_ids=batch.embed_ids, sentence_len=batch.sentence_len)
            pred_sentences_targets = self._get_targets(
                labels_indexes=logits.to('cpu').data.numpy())
            for internal_index, targets in enumerate(pred_sentences_targets):
                sentence_index = batch.sentence_index[internal_index]
                sentences[sentence_index].targets = targets
        return sentences

    def _get_targets(self, labels_indexes: np.array) -> List[List[Target]]:
        targets = []
        for sentence_labels in labels_indexes:
            sentence_targets = []
            labels = [x[0] for x in np.argwhere(sentence_labels > self.threshold)]
            for label in labels:
                sentence_targets.append(Target(nodes=[], category=self.aspect_labels[label]))
            targets.append(sentence_targets)
        return targets

    def save_model(self):
        """Save model state."""
        th.save(
            {
                'vocabulary': self.vocabulary,
                'model': self.model.state_dict(),
                'threshold': self.threshold,
            }, sentence_aspect_classifier_dump_path)

    @staticmethod
    def load_model() -> 'AspectClassifier':
        """Load pretrained state of model."""
        checkpoint = th.load(sentence_aspect_classifier_dump_path)
        model = checkpoint['model']
        classifier = AspectClassifier(vocabulary=checkpoint['vocabulary'],
                                      emb_matrix=model['nn.embed.weight'],
                                      threshold=checkpoint['threshold'])
        classifier.model.load_state_dict(model)
        return classifier

    @staticmethod
    def score(sentences: List[ParsedSentence], sentences_pred: List[ParsedSentence]):
        """Score function of classifier.

        Match score for task 5 sb 1.1 of SemEval2016.
        """
        total_labels = 0
        total_predictions = 0
        correct_predictions = 0

        for sentence_index in range(len(sentences)):
            categories = set([t.category for t in sentences[sentence_index].targets])
            pred_categories = set([t.category for t in sentences_pred[sentence_index].targets])

            correct_predictions += len(categories.intersection(pred_categories))
            total_labels += len(categories)
            total_predictions += len(pred_categories)

        if total_predictions == 0:
            return Score(precision=1.0, recall=0.0, f1=0.0)
        if correct_predictions == 0:
            return Score(precision=0.0, recall=0.0, f1=0.0)
        precision = correct_predictions / total_predictions
        recall = correct_predictions / total_labels
        f1 = 2 * (precision * recall) / (precision + recall)
        return Score(precision=precision, recall=recall, f1=f1)


@dataclass
class Score:
    precision: float
    recall: float
    f1: float
