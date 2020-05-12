from typing import List, Dict
import logging
import sys
import copy
from functools import reduce
from itertools import chain

import scipy
import torch as th
import numpy as np
from gensim.models import KeyedVectors
from frozendict import frozendict
from tqdm import tqdm

from src import SCORE_DECIMAL_LEN, PROGRESSBAR_COLUMNS_NUM, aspect_classifier_dump_path
from src.review.parsed_sentence import ParsedSentence
from src.review.target import Target
from .loader import DataLoader, Batch
from .nn.classifier import NNWrap
from .labels import ASPECT_LABELS


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
                 aspect_labels=ASPECT_LABELS):
        self.device = th.device('cpu')
        #self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.aspect_labels = aspect_labels
        num_labels = aspect_labels.shape[0]

        # prepare vocabulary and embeddings
        if isinstance(word2vec, KeyedVectors):
            self.vocabulary = {w: i for i, w in enumerate(word2vec.index2word)}
            emb_matrix = th.FloatTensor(word2vec.vectors)
        else:
            self.vocabulary = vocabulary

        while True:
            try:
                self.model = NNWrap(emb_matrix=emb_matrix,
                                    device=self.device,
                                    num_class=num_labels)
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
        self.thresholds = np.array([0.5] * num_labels)

    def batch_metrics(self, batch: Batch):
        """Make a forward step and return metrics"""
        logits = self.model(embed_ids=batch.embed_ids,
                            graph=batch.graph,
                            sentence_len=batch.sentence_len)
        loss = th.nn.BCEWithLogitsLoss(logits, batch.polarity)
        corrects = (th.max(logits, 1)[1].view(
            batch.polarity.size()).data == batch.polarity.data).sum()
        acc = np.float(corrects) / batch.polarity.size()[0]
        return logits, loss, acc

    def get_targets(self, logits: th.Tensor, sentence_len: th.Tensor) -> List[List[Target]]:
        targets = []
        for l in th.split(logits, [x for x in sentence_len]):
            targets.append(self.get_target(logits=l.view((-1, ))))
        return targets

    def get_target(self, logits: th.Tensor) -> List[Target]:
        targets = []

        term_indexes = (logits > th.tensor(self.thresholds)).nonzero()
        if term_indexes:
            # current target
            target_terms = []
            target_aspects_id = set()
            # current word
            term, _ = term_indexes[0]
            aspects_id = set()
            for word_id, label_id in chain(term_indexes, (None, None)):
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
                        target_aspects = aspects_id

                    # next word
                    term = word_id
                    aspects = set([label_id])
        return targets

    def fit(self,
            train_sentences: List[ParsedSentence],
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
                                   device=self.device,
                                   aspect_labels=self.aspect_labels)

        with tqdm(total=num_epoch, ncols=PROGRESSBAR_COLUMNS_NUM,
                  file=sys.stdout) as progress_bar:
            for epoch in range(num_epoch):
                # Train
                self.model.train()
                for i, batch in enumerate(train_batches):
                    optimizer.zero_grad()
                    logits = self.model(embed_ids=batch.embed_ids,
                                        graph=batch.graph,
                                        sentence_len=batch.sentence_len)
                    loss = self.criterion(logits, batch.labels)
                    loss.backward()
                    optimizer.step()

                # Optimal threshold
                self.model.eval()
                logits = []
                for i, batch in enumerate(train_batches):
                    logit = self.model(embed_ids=batch.embed_ids,
                                       graph=batch.graph,
                                       sentence_len=batch.sentence_len)
                    s_len = batch.sentence_len.to('cpu')
                    logits.append(logit.to('cpu'))
                logits = th.cat(logits, dim=0)
                sentence_len = th.cat(s_len, dim=0)
                targets_pred = self.get_targets(logits=logits, sentence_len=sentence_len)
                min_params = scipy.optimize.fmin(func=lambda thresholds: self._targets_score(
                    targets=[sentence.targets for sentence in train_sentences],
                    targets_pred=targets_pred,
                    thresholds=thresholds),
                                                 x0=self.thresholds)
                self.thresholds = min_params[0]
                logging.info(f'{min_params[1]}')

                progress_bar.update(1)

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
        batches = DataLoader(sentences=sentences,
                             batch_size=self.batch_size,
                             vocabulary=self.vocabulary,
                             device=self.device,
                             aspect_labels=self.aspect_labels)
        for batch_index, batch in enumerate(batches):
            logits, gcn_outputs = self.model(embed_ids=batch.embed_ids,
                                             graph=batch.graph,
                                             target_mask=batch.target_mask,
                                             sentence_len=batch.sentence_len)
            pred_labels = np.argmax(logits.data.numpy(), axis=1).tolist()
            for item_index, item in enumerate(pred_labels):
                sentence_index = batch.sentence_index[item_index]
                target_index = batch.target_index[item_index]
                sentences[sentence_index].targets[target_index].set_polarity(item)
        return sentences

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

    def _targets_score(self, targets: List[List[Target]], targets_pred: List[List[Target]],
                       thresholds: None) -> float:
        if thresholds:
            self.thresholds = thresholds

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

        precision = correct_predictions / total_targets
        recall = correct_predictions / total_predictions
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

    @staticmethod
    def score(sentences: List[ParsedSentence], sentences_pred: List[ParsedSentence]):
        """Return F1 score"""
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
        return precision, recall, f1
