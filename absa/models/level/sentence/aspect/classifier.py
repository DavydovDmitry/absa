import copy
import logging
import pathlib
import sys
from typing import List, Dict, Union, Tuple

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
from absa.models.level.sentence.aspect.nn import NeuralNetwork
from absa.models.score.f1 import Score

VERBOSITY_ON, VERBOSITY_PROGRESS, VERBOSITY_OFF = 'verbose', 'progress_bar', 'silence'


class AspectClassifier(BaseEstimator):
    """Sentence-level aspect classifier"""
    def __init__(
            self,
            batch_size: int = 100,
            layers_dim: np.array = np.array([40]),
            emb_dropout: float = 0.7,
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
        self.layers_dim = layers_dim
        self.emb_dropout = emb_dropout

        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')

    def _load_model(self, vocabulary: Dict[str, int], aspect_labels: Labels,
                    threshold: np.array, embeddings: th.Tensor):
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
        self.threshold_ = threshold

        while True:
            try:
                self.model = NeuralNetwork(embeddings=embeddings,
                                           num_classes=len(self.aspect_labels),
                                           device=self.device,
                                           layers_dim=self.layers_dim,
                                           emb_dropout=self.emb_dropout).to(self.device)
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
        start_threshold: np.array = None,
        fixed_threshold: bool = False,
        save_state: bool = False,
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

        start_threshold : np.array
            start value of threshold
        fixed_threshold : bool
            optimize threshold during training or set it immutable

        save_state : bool
        verbose : str
            the level of details in logging
        """

        aspect_labels = Labels(ASPECT_LABELS)
        if start_threshold is None:
            start_threshold = np.random.random(len(aspect_labels)) - 1.0
        self._load_model(vocabulary=vocabulary,
                         embeddings=embeddings,
                         aspect_labels=aspect_labels,
                         threshold=start_threshold)

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
                opt_results = minimize(lambda threshold: -self._opt_score(
                    logits=train_logits, labels=train_labels, threshold=threshold),
                                       x0=self.threshold_)
                self.threshold_ = opt_results.x
                f1_score = -opt_results.fun
            else:
                f1_score = self._opt_score(logits=train_logits,
                                           labels=train_labels,
                                           threshold=self.threshold_)
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
                val_f1_history[epoch] = self._opt_score(logits=val_logits,
                                                        labels=val_labels,
                                                        threshold=self.threshold_)
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

    def _logits2opinions(self, labels_indexes: np.array) -> List[List[Opinion]]:
        opinions = []
        for sentence_labels in labels_indexes:
            sentence_opinions = []
            labels = [x[0] for x in np.argwhere(sentence_labels > self.threshold_)]
            for label in labels:
                sentence_opinions.append(Opinion(nodes=[], category=self.aspect_labels[label]))
            opinions.append(sentence_opinions)
        return opinions

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
            pred_sentences_targets = self._logits2opinions(
                labels_indexes=logits.to('cpu').data.numpy())
            for internal_index, opinions in enumerate(pred_sentences_targets):
                text_index = batch.text_index[internal_index]
                sentence_index = batch.sentence_index[internal_index]
                texts[text_index].sentences[sentence_index].opinions = opinions
        return texts

    @staticmethod
    def _opt_score(logits: np.array, labels: np.array, threshold: np.array) -> float:
        """Calculate f1 score directly from nn output.

        Parameters
        ----------
        logits : np.array
            neural network predictions for every aspect.
        labels : np.array
            array of {0, 1} where:
            - 1 - aspect present in sentence;
            - 0 - otherwise.
        threshold : np.array
            threshold for aspect selection.

        Returns
        -------
        f1_score : float
            macro f1 score
        """

        labels_pred = np.where(logits > threshold, 1, 0)
        correct_predictions = labels[labels_pred.nonzero()].sum()
        total_predictions = labels_pred.sum()
        total_labels = labels.sum()

        if (correct_predictions == 0) or (total_predictions == 0):
            return 0.0
        precision = correct_predictions / total_predictions
        recall = correct_predictions / total_labels
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

    def get_metrics(self, texts: List[ParsedText], texts_pred: List[ParsedText]) -> Score:
        """Make predictions and return metrics

        Parameters
        ----------
        texts : List[Parsed]

        Returns
        -------
        score : Score
        """

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
        elif correct_predictions == 0:
            return Score(precision=0.0, recall=0.0, f1=0.0)
        precision = correct_predictions / total_predictions
        recall = correct_predictions / total_labels
        f1 = 2 * (precision * recall) / (precision + recall)
        return Score(precision=precision, recall=recall, f1=f1)

    def score(self, X: List[ParsedText]) -> float:
        """Make predictions and return f1 metric

        Parameters
        ----------
        X : List[ParsedText]
            text to be classified

        Returns
        -------
        f1_score : float
            macro f1 metric
        """
        X_pred = self.predict(X)
        return self.get_metrics(X, X_pred).f1

    def fit_predict_score(self, vocabulary, emb_matrix, train_reviews: List[ParsedText],
                          test_reviews: List[ParsedText], **kwargs) -> List[ParsedText]:
        """Fit model, return predictions, log score"""

        self.fit(train_texts=train_reviews,
                 vocabulary=vocabulary,
                 embeddings=emb_matrix,
                 **kwargs)
        test_reviews_pred = copy.deepcopy(test_reviews)
        for review in test_reviews_pred:
            review.reset_opinions()
        test_reviews_pred = self.predict(test_reviews_pred)
        logging.info(f'F1: {self.get_metrics(test_reviews, test_reviews_pred).f1}')
        return test_reviews_pred

    def save_model(self, pathway: pathlib.Path = sentence_aspect_classifier_dump_path) -> None:
        """Save model state

        Model state:
        - classifier hyper-parameters (that was passed to __init__())
        - neural network state

        Parameters
        ----------
        pathway
            pathway where classifier will be saved
        """
        pathway.parent.mkdir(parents=True, exist_ok=True)
        th.save(
            {
                'params': self.get_params(),
                'aspect_labels': self.aspect_labels,
                'vocabulary': self.vocabulary,
                'threshold': self.threshold_,
                'model': self.model.state_dict(),
            }, pathway)

    @staticmethod
    def load_model(pathway=sentence_aspect_classifier_dump_path) -> 'AspectClassifier':
        """Get model state

        Parameters
        ----------
        pathway
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
                               embeddings=model['embeddings.weight'])
        classifier.model.load_state_dict(model)
        return classifier
