from typing import List, Dict
import logging
import sys
import time
import copy
from functools import reduce
from itertools import chain
from dataclasses import dataclass

import scipy
import torch as th
import numpy as np
from gensim.models import KeyedVectors
from frozendict import frozendict
from tqdm import tqdm
from sklearn.metrics import f1_score

from src import PROGRESSBAR_COLUMNS_NUM, aspect_classifier_dump_path, SCORE_DECIMAL_LEN
from src.review.parsed_sentence import ParsedSentence
from src.review.target import Target
from .loader import DataLoader, Batch
from .nn import NeuralNetwork
from .labels import ASPECT_LABELS, Labels


class AspectClassifier:
    """Polarity classifier

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
                 word2vec: KeyedVectors = ...,
                 vocabulary: Dict[str, int] = ...,
                 emb_matrix: th.Tensor = ...,
                 aspect_labels=ASPECT_LABELS,
                 batch_size=100):
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.aspect_labels = Labels(labels=aspect_labels, none_value='NONE_VALUE')

        # prepare vocabulary and embeddings
        if isinstance(word2vec, KeyedVectors):
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

    def fit(self,
            train_sentences: List[ParsedSentence],
            val_sentences=None,
            optimizer_class=th.optim.Adamax,
            optimizer_params=frozendict({
                'lr': 0.01,
                'weight_decay': 0,
            }),
            num_epoch=30,
            verbose=True,
            save_state=True):
        """Fit on train sentences and save model state."""
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        if verbose:
            logging.info(
                f'Number of parameters: {sum((reduce(lambda x, y: x * y, p.shape)) for p in parameters)}'
            )

        optimizer = optimizer_class(parameters, **optimizer_params)

        train_batches = DataLoader(sentences=train_sentences,
                                   batch_size=self.batch_size,
                                   vocabulary=self.vocabulary,
                                   aspect_labels=self.aspect_labels,
                                   device=self.device)
        if val_sentences:
            val_batches = DataLoader(sentences=val_sentences,
                                     batch_size=self.batch_size,
                                     vocabulary=self.vocabulary,
                                     aspect_labels=self.aspect_labels,
                                     device=self.device)
            val_acc_history, val_loss_history, f1_history = [], [], []

        train_acc_history, train_loss_history = [], []

        for epoch in range(num_epoch):

            # Train
            start_time = time.process_time()
            train_len = len(train_batches)
            self.model.train()
            train_loss, train_acc = 0., 0.
            for i, batch in enumerate(train_batches):
                optimizer.zero_grad()
                logits = self.model(embed_ids=batch.embed_ids,
                                    graph=batch.graph,
                                    sentence_len=batch.sentence_len)
                loss = th.nn.functional.cross_entropy(logits, batch.labels, reduction='mean')
                loss.backward()
                optimizer.step()

                train_loss += loss.data

            # Validation
            if val_sentences:
                val_len = len(val_batches)
                self.model.eval()
                predictions, labels = [], []
                val_loss, val_acc = 0., 0.
                for i, batch in enumerate(val_batches):
                    logits = self.model(embed_ids=batch.embed_ids,
                                        graph=batch.graph,
                                        sentence_len=batch.sentence_len)
                    loss = th.nn.functional.cross_entropy(logits,
                                                          batch.labels,
                                                          reduction='mean')
                    val_loss += loss.data
                    predictions += np.argmax(logits.to('cpu').data.numpy(), axis=1).tolist()
                    labels += batch.labels.to('cpu').data.numpy().tolist()
                f1 = f1_score(labels, predictions, average='macro')

                train_loss = train_loss / train_len
                val_loss = val_loss / val_len

                logging.info('-' * 40 + f' Epoch {epoch:03d} ' + '-' * 40)
                logging.info(f'Elapsed time: {(time.process_time() - start_time):.{3}f} sec')
                logging.info(f'Train      ' + f'loss: {train_loss:.{SCORE_DECIMAL_LEN}f}| ')
                logging.info(f'Validation ' + f'loss: {(val_loss):.{SCORE_DECIMAL_LEN}f}| ' +
                             f'f1_score: {f1:.{SCORE_DECIMAL_LEN}f}')

                val_loss_history.append(val_loss)
                f1_history.append(f1)
            train_loss_history.append(train_loss)

        if save_state:
            self.save_model()

    def predict(self, sentences: List[ParsedSentence]) -> List[ParsedSentence]:
        """Modify passed sentences. Define every target polarity.

        Parameters
        ----------
        sentences : List[ParsedSentence]
            Sentences with extracted targets.

        Return
        -------
        sentences : List[ParsedSentence]
            Sentences with defined polarity of every target.
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
            logits = self.model(embed_ids=batch.embed_ids,
                                graph=batch.graph,
                                sentence_len=batch.sentence_len)
            pred_labels = th.argmax(logits.to('cpu'), dim=1)
            pred_sentences_targets = self._get_targets(
                labels_indexes=pred_labels,
                sentence_len=[x.item() for x in batch.sentence_len.to('cpu')])
            for internal_index, targets in enumerate(pred_sentences_targets):
                sentence_index = batch.sentence_index[internal_index]
                sentence_nodes = sentences[sentence_index].get_sentence_order()
                for target in targets:
                    target.nodes = [sentence_nodes[x] for x in target.nodes]
                    sentences[sentence_index].targets.append(target)
        return sentences

    def _get_targets(self, labels_indexes: th.Tensor,
                     sentence_len: List[int]) -> List[List[Target]]:
        targets = []
        for indexes in th.split(labels_indexes, sentence_len):
            targets.append(self._get_target(indexes.data.numpy()))
        return targets

    def _get_target(self, labels_indexes: np.array) -> List[Target]:
        targets = []

        words_indexes = [
            x[0] for x in np.argwhere(
                labels_indexes != self.aspect_labels.get_index(self.aspect_labels.none_value))
        ]
        if words_indexes:
            target_words = [words_indexes[0]]
            target_label_index = labels_indexes[target_words[0]]

            for word_index in words_indexes:
                label_index = labels_indexes[word_index]
                if (word_index - target_words[-1] == 1) and (label_index
                                                             == target_label_index):
                    target_words.append(word_index)
                else:
                    targets.append(
                        Target(nodes=target_words,
                               category=self.aspect_labels[target_label_index]))
                    target_words = [word_index]
                    target_label_index = label_index
            if target_words:  # always
                targets.append(
                    Target(nodes=target_words,
                           category=self.aspect_labels[target_label_index]))

        return targets

    def save_model(self):
        """Save model state."""
        th.save({
            'vocabulary': self.vocabulary,
            'model': self.model.state_dict()
        }, aspect_classifier_dump_path)

    @staticmethod
    def load_model() -> 'AspectClassifier':
        """Load pretrained model."""
        checkpoint = th.load(aspect_classifier_dump_path)
        model = checkpoint['model']
        classifier = AspectClassifier(vocabulary=checkpoint['vocabulary'],
                                      emb_matrix=model['nn.embed.weight'])
        classifier.model.load_state_dict(model)
        return classifier

    @staticmethod
    def score(sentences: List[ParsedSentence], sentences_pred: List[ParsedSentence]):
        total_targets = 0
        total_predictions = 0
        correct_predictions = 0

        for sentence_index in range(len(sentences)):
            for y in sentences[sentence_index].targets:
                for y_pred in sentences_pred[sentence_index].targets:
                    if (y.nodes == y_pred.nodes) and (y.category == y_pred.category):
                        correct_predictions += 1
                        break
            # total_targets += len([t for t in sentences[sentence_index].targets if t.nodes])
            total_targets += len(sentences[sentence_index].targets)
            total_predictions += len(sentences_pred[sentence_index].targets)

        if total_predictions == 0:
            return Score(precision=1.0, recall=0.0, f1=0.0)
        if correct_predictions == 0:
            return Score(precision=0.0, recall=0.0, f1=0.0)
        precision = correct_predictions / total_targets
        recall = correct_predictions / total_predictions
        f1 = 2 * (precision * recall) / (precision + recall)
        return Score(precision=precision, recall=recall, f1=f1)


@dataclass
class Score:
    precision: float
    recall: float
    f1: float
