from typing import List, Dict, Union, Tuple
import copy
import sys

import torch as th
import numpy as np
from frozendict import frozendict
from scipy.optimize import minimize as minimize
from sklearn.base import BaseEstimator
from tqdm import tqdm

from absa import sentence_aspect_classifier_dump_path, PROGRESSBAR_COLUMNS_NUM
from absa.text.parsed.text import ParsedText
from absa.text.parsed.opinion import Opinion
from absa.labels.labels import Labels
from absa.labels.default import ASPECT_LABELS
from .loader import DataLoader
from .nn.nn import NeuralNetwork
from ...score.f1 import Score

VERBOSITY_ON, VERBOSITY_PROGRESS, VERBOSITY_OFF = 'verbose', 'progress_bar', 'silence'


class AspectClassifier(BaseEstimator):
    """Sentence-level aspect classifier"""
    def __init__(
            self,
            batch_size: int = 100,
            nn_params: Dict = frozendict({
                'layers_dim': np.array([40]),
            }),
            optimizer_class=th.optim.Adam,
            optimizer_params: Dict = frozendict({
                'lr': 0.01,
            }),
            num_epoch=50,
    ):
        """Set classifier hyper-parameters

        Parameters
        ----------
        """
        self.batch_size = batch_size
        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params
        self.num_epoch = num_epoch
        self.nn_params = nn_params

        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')

    def _load_model(self, vocabulary: Dict[str, int], aspect_labels: Labels,
                    threshold: np.array, embeddings: th.Tensor) -> None:
        """Set classifier parameters

        Parameters
        ----------
        vocabulary : Dict[str, int]
            [word] -> it's index
        embeddings : th.Tensor
            tensor of word's embeddings

        aspect_labels : np.array
            collection of labels
        threshold : np.array
            threshold for labels
        """

        self.vocabulary = vocabulary
        self.aspect_labels = aspect_labels
        self.threshold = threshold

        while True:
            try:
                self.model = NeuralNetwork(embeddings=embeddings,
                                           num_classes=len(self.aspect_labels),
                                           device=self.device,
                                           **self.nn_params).to(self.device)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    th.cuda.empty_cache()
            else:
                break

    def fit(
        self,
        train_texts: List[ParsedText],
        vocabulary: Dict[str, int],
        embeddings: th.Tensor,
        val_texts: List[ParsedText] = None,
        init_threshold: np.array = None,
        fixed_threshold: bool = False,
        save_state: bool = True,
        verbose: Union[VERBOSITY_ON, VERBOSITY_PROGRESS, VERBOSITY_OFF] = VERBOSITY_PROGRESS
    ) -> Union[np.array, Tuple[np.array, np.array]]:
        """Fit and save model state.

        Every epoch consist from 2 stages
        - Train
            Optimize parameters of neural network and select optimal
            threshold for every class.
        - Validation
            Calculate scores on unseen texts.

        Parameters
        ----------
        train_texts : List[ParsedText]
            train sentence
        val_texts : List[ParsedText]
            validation sentences

        vocabulary : dict
            dictionary where key is wordlemma_POS value - index in embedding matrix
        embeddings : th.Tensor
            matrix of pretrained embeddings

        init_threshold : np.array
            start value of threshold
        fixed_threshold : bool
            optimize threshold during training or set it immutable

        save_state : bool
        verbose : str
            the level of details in logging
        """

        aspect_labels = Labels(ASPECT_LABELS)
        if init_threshold is None:
            init_threshold = np.random.random(len(aspect_labels)) - 1.0
        self._load_model(vocabulary=vocabulary,
                         embeddings=embeddings,
                         aspect_labels=aspect_labels,
                         threshold=init_threshold)

        loss_criterion = th.nn.BCEWithLogitsLoss()
        loss_func = lambda y_pred, y: loss_criterion(y_pred, y)

        # optimized parameters
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = self.optimizer_class(parameters, **self.optimizer_params)

        train_batches = DataLoader(texts=train_texts,
                                   batch_size=self.batch_size,
                                   vocabulary=self.vocabulary,
                                   aspect_labels=self.aspect_labels,
                                   device=self.device)
        train_f1_history = np.empty(shape=(self.num_epoch, ), dtype=np.float)

        if val_texts:
            val_batches = DataLoader(texts=val_texts,
                                     batch_size=self.batch_size,
                                     vocabulary=self.vocabulary,
                                     aspect_labels=self.aspect_labels,
                                     device=self.device)
            val_f1_history = np.empty(shape=(self.num_epoch, ), dtype=np.float)

        def epoch_step():
            # Train
            self.model.train()
            train_logits, train_labels = [], []
            for i, batch in enumerate(train_batches):
                optimizer.zero_grad()
                logits = self.model(embed_ids=batch.embed_ids, sentence_len=batch.sentence_len)
                loss = loss_func(logits, batch.labels)
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
            if val_texts:
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
                if verbose == VERBOSITY_ON:
                    pass
                    # todo: logging

        # train cycle
        if verbose == VERBOSITY_PROGRESS:
            with tqdm(total=self.num_epoch, ncols=PROGRESSBAR_COLUMNS_NUM,
                      file=sys.stdout) as progress_bar:
                for epoch in range(self.num_epoch):
                    epoch_step()
                    progress_bar.update(1)
        else:
            for epoch in range(self.num_epoch):
                epoch_step()

        if save_state:
            self.save_model()

        if val_texts is not None:
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

    def predict(self, texts: List[ParsedText]) -> List[ParsedText]:
        """Modify passed sentences. Add targets with empty list of nodes.

        Parameters
        ----------
        texts : List[ParsedText]
            Sentences with extracted targets.

        Return
        -------
        texts : List[ParsedReview]
            Sentences with defined targets.
        """
        def _get_opinions(labels_indexes: np.array) -> List[List[Opinion]]:
            opinions = []
            for sentence_labels in labels_indexes:
                sentence_opinions = []
                labels = [x[0] for x in np.argwhere(sentence_labels > self.threshold)]
                for label in labels:
                    sentence_opinions.append(
                        Opinion(nodes=[], category=self.aspect_labels[label]))
                opinions.append(sentence_opinions)
            return opinions

        self.model.eval()
        texts = copy.deepcopy(texts)
        for text in texts:
            text.reset_opinions()

        batches = DataLoader(texts=texts,
                             batch_size=self.batch_size,
                             vocabulary=self.vocabulary,
                             aspect_labels=self.aspect_labels,
                             device=self.device)
        for batch_index, batch in enumerate(batches):
            logits = self.model(embed_ids=batch.embed_ids, sentence_len=batch.sentence_len)
            pred_sentences_targets = _get_opinions(
                labels_indexes=logits.to('cpu').data.numpy())
            for internal_index, opinions in enumerate(pred_sentences_targets):
                text_index = batch.text_index[internal_index]
                sentence_index = batch.sentence_index[internal_index]
                texts[text_index].sentences[sentence_index].opinions = opinions
        return texts

    def get_scores(self, texts: List[ParsedText]) -> Score:
        """Macro classification metrics.

        Parameters
        ----------
        texts : List[Parsed]

        Returns
        -------
        score : Score
        """

        texts_pred = self.predict(texts)

        total_labels = 0
        total_predictions = 0
        correct_predictions = 0

        for text, text_pred in zip(texts, texts_pred):
            for sentence, sentence_pred in zip(text, text_pred):
                categories = set([t.category for t in sentence.opinions])
                pred_categories = set([t.category for t in sentence_pred.opinions])

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

    def score(self, X: List[ParsedText]) -> float:
        return self.get_scores(X).f1

    def save_model(self, pathway: str = sentence_aspect_classifier_dump_path) -> None:
        """Save model state

        Model state:
        - classifier hyper-parameters (that was passed to __init__())
        - neural network state

        Parameters
        ----------
        pathway : str
            pathway where classifier will be saved
        """
        th.save(
            {
                'params': self.get_params(),
                'aspect_labels': self.aspect_labels,
                'vocabulary': self.vocabulary,
                'threshold': self.threshold,
                'model': self.model.state_dict(),
            }, pathway)

    @staticmethod
    def load_model(pathway: str = sentence_aspect_classifier_dump_path) -> 'AspectClassifier':
        """Get model state

        Parameters
        ----------
        pathway : str
            pathway where classifier was saved

        Returns
        -------
        classifier : AspectClassifier
            Trained sentence-level aspect classifier
        """

        checkpoint = th.load(pathway)
        model = checkpoint['model']
        classifier = AspectClassifier(checkpoint['params'])
        classifier._load_model(aspect_labels=checkpoint['aspect_labels'],
                               vocabulary=checkpoint['vocabulary'],
                               threshold=checkpoint['threshold'],
                               embeddings=model['nn.embed.weight'])
        classifier.model.load_state_dict(model)
        return classifier
