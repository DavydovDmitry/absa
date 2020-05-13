from typing import List, Dict
import logging
import sys
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

from src import PROGRESSBAR_COLUMNS_NUM, aspect_classifier_dump_path
from src.review.parsed_sentence import ParsedSentence
from src.review.target import Target
from .loader import DataLoader, Batch
from .nn import NeuralNetwork
from .labels import ASPECT_LABELS


@dataclass
class Score:
    precision: float
    recall: float
    f1: float


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
    aspect_labels : np.array
        array of aspect categories
    """
    def __init__(self,
                 word2vec: KeyedVectors = ...,
                 vocabulary: Dict[str, int] = ...,
                 emb_matrix: th.Tensor = ...,
                 batch_size=100,
                 aspect_labels=ASPECT_LABELS,
                 thresholds=None):
        # self.device = th.device('cpu')
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.aspect_labels = aspect_labels
        self.num_labels = aspect_labels.shape[0]

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
                                           num_class=self.num_labels).to(self.device)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    # todo: handle
                    # for p in self.model.parameters():
                    #     if p.grad is not None:
                    #         del p.grad
                    th.cuda.empty_cache()
            else:
                break

        self.criterion = th.nn.BCEWithLogitsLoss()
        self.thresholds = None

    def get_targets(self, logits: th.Tensor, sentence_len: th.Tensor,
                    thresholds) -> List[List[Target]]:
        targets = []
        for l in th.split(logits, [x for x in sentence_len]):
            targets.append(
                self.get_target(logits=l.view((-1, self.num_labels)), thresholds=thresholds))
        return targets

    def get_target(self, logits: th.Tensor, thresholds: np.array) -> List[Target]:
        targets = []

        term_indexes = (logits > th.tensor(thresholds)).nonzero(as_tuple=False)
        if term_indexes.size(0):
            # current target
            target_terms = []
            target_aspects_id = set()
            # current word
            term = term_indexes[0][0].item()
            aspects_id = set()
            for word_id, label_id in chain([(x[0].item(), x[1].item()) for x in term_indexes],
                                           [(None, None)]):
                if word_id == term:
                    aspects_id.add(label_id)
                else:
                    # the same aspect term
                    if aspects_id == target_aspects_id:
                        target_terms.append(term)
                    # another aspect term
                    else:
                        for a in target_aspects_id:
                            targets.append(
                                Target(nodes=target_terms, category=self.aspect_labels[a]))
                        target_terms = [term]
                        target_aspects_id = aspects_id

                    # next word
                    term = word_id
                    aspects_id = set([label_id])
            if target_terms:
                for a in target_aspects_id:
                    targets.append(Target(nodes=target_terms, category=self.aspect_labels[a]))
        return targets

    def fit(self,
            train_sentences: List[ParsedSentence],
            optimizer_class=th.optim.Adam,
            optimizer_params=frozendict({
                'lr': 0.001,
                'weight_decay': 0,
            }),
            num_epoch=50,
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
                                   device=self.device,
                                   aspect_labels=self.aspect_labels)

        # with tqdm(total=num_epoch, ncols=PROGRESSBAR_COLUMNS_NUM,
        #           file=sys.stdout) as progress_bar:
        for epoch in range(num_epoch):
            # Train
            self.model.train()
            epoch_loss = 0.0
            for i, batch in enumerate(train_batches):
                optimizer.zero_grad()
                logits = self.model(embed_ids=batch.embed_ids,
                                    graph=batch.graph,
                                    sentence_len=batch.sentence_len)
                loss = self.criterion(logits, batch.labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss

            logging.info(f'Epoch: {epoch:03d} | Loss: {epoch_loss}')
            # progress_bar.update(1)

        if save_state:
            self.save_model()

    def select_threshold(self, sentences):
        # x0 = np.array([
        #     0.01008897, 0.0118017, 0.1080818, 0.04046877, 0.03043046, 0.0027503, 0.03002802,
        #     0.02972298, 0.02950508, 0.0290992, 0.02938878, 0.02900143
        # ])
        x0 = np.array([
            0.01, 0.0001, 0.0001, 0.04, 0.002, 0.04633, 0.3237, 0.02, 0.04633, 0.0001, 0.0463,
            0.04633
        ])
        batches = DataLoader(sentences=sentences,
                             batch_size=self.batch_size,
                             vocabulary=self.vocabulary,
                             device=self.device,
                             aspect_labels=self.aspect_labels)

        self.model.eval()
        logits = []
        sentence_len = []
        for i, batch in enumerate(batches):
            logit = self.model(embed_ids=batch.embed_ids,
                               graph=batch.graph,
                               sentence_len=batch.sentence_len)
            logit = th.sigmoid(logit)
            sentence_len.append(batch.sentence_len.to('cpu'))
            logits.append(logit.to('cpu'))

        min_params = scipy.optimize.minimize(lambda thresholds: -self._targets_score(
            targets=[sentence.targets for sentence in sentences if len(sentence)],
            logits=th.cat(logits, dim=0),
            sentence_len=th.cat(sentence_len, dim=0),
            thresholds=thresholds),
                                             x0=x0)
        logging.info(f'{min_params.x}')
        self.thresholds = min_params.x

    def _targets_score(self, targets: List[List[Target]], logits: th.Tensor,
                       sentence_len: th.Tensor, thresholds: np.array) -> float:
        targets_pred = self.get_targets(logits=logits,
                                        sentence_len=sentence_len,
                                        thresholds=thresholds)

        total_targets = 0
        total_predictions = 0
        correct_predictions = 0

        for sentence_index in range(len(targets)):
            for y in targets[sentence_index]:
                for y_pred in targets_pred[sentence_index]:
                    if (y.nodes == y_pred.nodes) and (y.category == y_pred.category):
                        correct_predictions += 1
                        break
            total_targets += len(targets[sentence_index])
            total_predictions += len(targets_pred[sentence_index])

        if correct_predictions == 0:
            return 0.0
        precision = correct_predictions / total_targets
        recall = correct_predictions / total_predictions
        f1 = 2 * (precision * recall) / (precision + recall)
        logging.info({f'{f1}: {[x for x in thresholds]}'})
        return f1

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
        not_empty_sentence_indexes = [
            index for index, sentence in enumerate(sentences) if len(sentence)
        ]
        batches = DataLoader(sentences=sentences,
                             batch_size=self.batch_size,
                             vocabulary=self.vocabulary,
                             device=self.device,
                             aspect_labels=self.aspect_labels)
        for batch_index, batch in enumerate(batches):
            logits = self.model(embed_ids=batch.embed_ids,
                                graph=batch.graph,
                                sentence_len=batch.sentence_len)
            logits = th.sigmoid(logits)
            targets_pred = self.get_targets(logits=logits.to('cpu'),
                                            sentence_len=batch.sentence_len.to('cpu'),
                                            thresholds=self.thresholds)

            for sentence_index, targets in enumerate(targets_pred):
                sentences[not_empty_sentence_indexes[sentence_index]].targets = targets
        return sentences

    def save_model(self):
        """Save model state."""
        th.save(
            {
                'vocabulary': self.vocabulary,
                'model': self.model.state_dict(),
                'thresholds': self.thresholds
            }, aspect_classifier_dump_path)

    @staticmethod
    def load_model() -> 'AspectClassifier':
        """Load pretrained model."""
        checkpoint = th.load(aspect_classifier_dump_path)
        model = checkpoint['model']
        classifier = AspectClassifier(vocabulary=checkpoint['vocabulary'],
                                      emb_matrix=model['nn.embed.weight'],
                                      thresholds=checkpoint['thresholds'])
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
