from typing import List, Dict
import logging
import time
import copy
from functools import reduce
from dataclasses import dataclass

import torch as th
import numpy as np
from gensim.models import KeyedVectors
from frozendict import frozendict
from scipy.optimize import minimize as minimize

from src import sentence_aspect_classifier_dump_path, SCORE_DECIMAL_LEN
from src.review.parsed_sentence import ParsedSentence
from src.review.target import Target
from .loader import DataLoader
from .nn.nn import NeuralNetwork
from src.labels.labels import Labels
from src.labels.default import ASPECT_LABELS


class AspectClassifier:
    """Aspect classifier

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
        self.aspect_labels = Labels(labels=aspect_labels)

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

        criterion = th.nn.BCEWithLogitsLoss()
        self.loss_func = lambda y_pred, y: criterion(y_pred, y)

    def fit(self,
            train_sentences: List[ParsedSentence],
            optimizer_class=th.optim.Adam,
            optimizer_params=frozendict({
                'lr': 0.01,
            }),
            num_epoch=50,
            verbose=True,
            save_state=True):
        """Fit on train sentences and save model state."""
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        if verbose:
            logging.info(
                f'Number of trainable parameters: {sum((reduce(lambda x, y: x * y, p.shape)) for p in parameters)}'
            )

        optimizer = optimizer_class(parameters, **optimizer_params)

        train_batches = DataLoader(sentences=train_sentences,
                                   batch_size=self.batch_size,
                                   vocabulary=self.vocabulary,
                                   aspect_labels=self.aspect_labels,
                                   device=self.device)

        for epoch in range(num_epoch):

            # Train
            start_time = time.process_time()
            train_len = len(train_batches)
            self.model.train()
            train_loss, train_acc = 0., 0.
            for i, batch in enumerate(train_batches):
                optimizer.zero_grad()
                logits = self.model(embed_ids=batch.embed_ids, sentence_len=batch.sentence_len)
                loss = self.loss_func(logits, batch.labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.data
            train_loss = train_loss / train_len

            logging.info('-' * 40 + f' Epoch {epoch:03d} ' + '-' * 40)
            logging.info(f'Elapsed time: {(time.process_time() - start_time):.{3}f} sec')
            logging.info(f'Train      ' + f'loss: {train_loss:.{SCORE_DECIMAL_LEN}f}| ')

        if save_state:
            self.save_model()

    def select_threshold(self, sentences: List[ParsedSentence]):
        def f1_score(threshold):
            labels_pred = np.where(logits > threshold, 1, 0)
            correct = labels[labels_pred.nonzero()].sum()
            total_predictions = labels_pred.sum()
            total_labels = labels.sum()

            precision = correct / total_predictions
            recall = correct / total_labels
            f1 = 2 * (precision * recall) / (precision + recall)
            # logging.info(f'{f1}: {threshold}')
            return -f1

        train_batches = DataLoader(sentences=sentences,
                                   batch_size=self.batch_size,
                                   vocabulary=self.vocabulary,
                                   aspect_labels=self.aspect_labels,
                                   device=self.device)

        logits = []
        labels = []
        self.model.eval()
        for i, batch in enumerate(train_batches):
            logit = self.model(embed_ids=batch.embed_ids,
                               sentence_len=batch.sentence_len).to('cpu').data.numpy()
            label = batch.labels.to('cpu').data.numpy()
            logits.append(logit)
            labels.append(label)
        logits = np.concatenate(logits)
        labels = np.concatenate(labels)
        x0 = np.random.random(len(self.aspect_labels)) - 1.0
        opt_results = minimize(f1_score, x0=x0, options={
            'adaptive': True,
        })
        logging.info(opt_results)
        self.threshold = opt_results.x

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
        th.save({
            'vocabulary': self.vocabulary,
            'model': self.model.state_dict()
        }, sentence_aspect_classifier_dump_path)

    @staticmethod
    def load_model() -> 'AspectClassifier':
        """Load pretrained model."""
        checkpoint = th.load(sentence_aspect_classifier_dump_path)
        model = checkpoint['model']
        classifier = AspectClassifier(vocabulary=checkpoint['vocabulary'],
                                      emb_matrix=model['nn.embed.weight'])
        classifier.model.load_state_dict(model)
        return classifier

    @staticmethod
    def score(sentences: List[ParsedSentence], sentences_pred: List[ParsedSentence]):
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
