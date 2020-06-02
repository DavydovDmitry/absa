from typing import List, Dict, Tuple, Union
import logging
import time
import copy
import sys

import torch as th
import numpy as np
from frozendict import frozendict
from sklearn.metrics import f1_score
from tqdm import tqdm

from absa import opinion_aspect_classifier_dump_path, SCORE_DECIMAL_LEN, PROGRESSBAR_COLUMNS_NUM
from absa.text.parsed.review import ParsedReview
from absa.text.parsed.sentence import ParsedSentence
from absa.text.opinion.opinion import Opinion
from absa.labels.labels import Labels
from absa.labels.default import ASPECT_LABELS
from .loader import DataLoader
from .nn.nn import NeuralNetwork
from ...score.f1 import Score


class AspectClassifier:
    """Target level aspect classifier

    Attributes
    ----------
    vocabulary : dict
        dictionary where key is wordlemma_POS value - index in embedding matrix
    emb_matrix : th.Tensor
        matrix of pretrained embeddings
    """
    def __init__(self,
                 vocabulary: Dict[str, int] = None,
                 emb_matrix: th.Tensor = None,
                 aspect_labels=ASPECT_LABELS,
                 batch_size=100):
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.aspect_labels = Labels(labels=aspect_labels, none_value='NONE_VALUE')

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
            train_texts: List[ParsedReview],
            val_texts=None,
            optimizer_class=th.optim.Adam,
            optimizer_params=frozendict({
                'lr': 0.01,
            }),
            num_epoch=50,
            verbose=False,
            save_state=True) -> Union[np.array, Tuple[np.array, np.array]]:
        """Fit on train sentences and save model state."""
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = optimizer_class(parameters, **optimizer_params)

        train_batches = DataLoader(texts=train_texts,
                                   batch_size=self.batch_size,
                                   vocabulary=self.vocabulary,
                                   aspect_labels=self.aspect_labels,
                                   device=self.device)
        if val_texts:
            val_batches = DataLoader(texts=val_texts,
                                     batch_size=self.batch_size,
                                     vocabulary=self.vocabulary,
                                     aspect_labels=self.aspect_labels,
                                     device=self.device)
            val_loss_history, val_f1_history = [], []
        train_f1_history, train_loss_history = [], []

        def epoch_step():

            # Train
            start_time = time.process_time()
            train_len = len(train_batches)
            self.model.train()
            train_predictions, train_labels = [], []
            train_loss, train_acc = 0., 0.
            for i, batch in enumerate(train_batches):
                optimizer.zero_grad()
                logits = self.model(embed_ids=batch.embed_ids, sentence_len=batch.sentence_len)
                loss = th.nn.functional.cross_entropy(logits, batch.labels, reduction='mean')
                loss.backward()
                optimizer.step()

                train_predictions += np.argmax(logits.to('cpu').data.numpy(), axis=1).tolist()
                train_labels += batch.labels.to('cpu').data.numpy().tolist()
                train_loss += loss.data
            train_f1 = f1_score(train_predictions, train_labels, average='macro')
            train_f1_history.append(train_f1)

            # Validation
            if val_texts:
                val_len = len(val_batches)
                self.model.eval()
                val_predictions, val_labels = [], []
                val_loss, val_acc = 0., 0.
                for i, batch in enumerate(val_batches):
                    logits = self.model(embed_ids=batch.embed_ids,
                                        sentence_len=batch.sentence_len)
                    loss = th.nn.functional.cross_entropy(logits,
                                                          batch.labels,
                                                          reduction='mean')
                    val_loss += loss.data
                    val_predictions += np.argmax(logits.to('cpu').data.numpy(),
                                                 axis=1).tolist()
                    val_labels += batch.labels.to('cpu').data.numpy().tolist()
                val_f1 = f1_score(val_labels, val_predictions, average='macro')

                train_loss = train_loss / train_len
                val_loss = val_loss / val_len

                if verbose:
                    logging.info('-' * 40 + f' Epoch {epoch:03d} ' + '-' * 40)
                    logging.info(
                        f'Elapsed time: {(time.process_time() - start_time):.{3}f} sec')
                    logging.info(f'Train      ' +
                                 f'loss: {train_loss:.{SCORE_DECIMAL_LEN}f}| ' +
                                 f'f1_score: {train_f1:.{SCORE_DECIMAL_LEN}f}')
                    logging.info(f'Validation ' +
                                 f'loss: {(val_loss):.{SCORE_DECIMAL_LEN}f}| ' +
                                 f'f1_score: {val_f1:.{SCORE_DECIMAL_LEN}f}')

                val_loss_history.append(val_loss)
                train_loss_history.append(train_loss)
                val_f1_history.append(val_f1)

        if verbose and (val_texts is not None):
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
        if val_texts:
            return train_f1_history, val_f1_history
        return train_f1_history

    def predict(self, texts: List[ParsedReview]) -> List[ParsedReview]:
        """Predict sentence aspect terms and it's categories.

        Expecting to receive sentences with defined sentence level aspect
        categories. If there is aspect category in sentence, then it has a
        implicit target (empty list of nodes) with this category.
        This function extend this list by explicit targets (non-empty list
        of nodes). If there is explicit target with category A then
        implicit target with the same category A will be removed.

        Parameters
        ----------
        texts : List[ParsedReviews]
            Sentences with extracted targets.

        Return
        -------
        sentences : List[ParsedSentence]
            Sentences with defined polarity of every target.
        """
        self.model.eval()
        texts = copy.deepcopy(texts)

        batches = DataLoader(texts=texts,
                             batch_size=self.batch_size,
                             vocabulary=self.vocabulary,
                             aspect_labels=self.aspect_labels,
                             device=self.device)
        for batch_index, batch in enumerate(batches):
            logits = self.model(embed_ids=batch.embed_ids, sentence_len=batch.sentence_len)
            pred_labels = th.argmax(logits.to('cpu'), dim=1)
            pred_sentences_opinions = self._get_opinions(
                labels_indexes=pred_labels,
                sentence_len=[x.item() for x in batch.sentence_len.to('cpu')])
            for internal_index, opinions in enumerate(pred_sentences_opinions):
                text_index = batch.text_index[internal_index]
                sentence_index = batch.sentence_index[internal_index]
                sentence_nodes = texts[text_index].sentences[
                    sentence_index].nodes_sentence_order()

                explicit_opinions = []
                explicit_categories = set()
                for opinion in opinions:
                    opinion.nodes = [sentence_nodes[x] for x in opinion.nodes]
                    explicit_opinions.append(opinion)
                    explicit_categories.add(opinion.category)
                for opinion in texts[text_index].sentences[sentence_index].opinions:
                    if (not opinion.nodes) and (opinion.category in explicit_categories):
                        texts[text_index].sentences[sentence_index].opinions.remove(opinion)
                texts[text_index].sentences[sentence_index].opinions.extend(explicit_opinions)
        return texts

    def _get_opinions(self, labels_indexes: th.Tensor,
                      sentence_len: List[int]) -> List[List[Opinion]]:
        """Convert predictions to targets

        Return
        ------
        targets : List[List[Opinion]]
            sentences and it's targets
        """

        opinions = []
        for indexes in th.split(labels_indexes, sentence_len):
            opinions.append(self._get_opinion(indexes.data.numpy()))
        return opinions

    def _get_opinion(self, labels_indexes: np.array) -> List[Opinion]:
        opinions = []

        words_indexes = [
            x[0] for x in np.argwhere(
                labels_indexes != self.aspect_labels.get_index(self.aspect_labels.none_value))
        ]
        if words_indexes:
            aspect_term = [words_indexes[0]]
            aspect_label_index = labels_indexes[aspect_term[0]]

            for word_index in words_indexes:
                label_index = labels_indexes[word_index]
                if (word_index - aspect_term[-1] == 1) and (label_index == aspect_label_index):
                    aspect_term.append(word_index)
                else:
                    opinions.append(
                        Opinion(nodes=aspect_term,
                                category=self.aspect_labels[aspect_label_index]))
                    aspect_term = [word_index]
                    aspect_label_index = label_index
            if aspect_term:  # always
                opinions.append(
                    Opinion(nodes=aspect_term,
                            category=self.aspect_labels[aspect_label_index]))

        return opinions

    def save_model(self):
        """Save model state."""
        th.save({
            'vocabulary': self.vocabulary,
            'model': self.model.state_dict()
        }, opinion_aspect_classifier_dump_path)

    @staticmethod
    def load_model() -> 'AspectClassifier':
        """Load pretrained model."""
        checkpoint = th.load(opinion_aspect_classifier_dump_path)
        model = checkpoint['model']
        classifier = AspectClassifier(vocabulary=checkpoint['vocabulary'],
                                      emb_matrix=model['nn.embed.weight'])
        classifier.model.load_state_dict(model)
        return classifier

    @staticmethod
    def score(texts: List[ParsedReview], texts_pred: List[ParsedReview]) -> Score:
        total_targets = 0
        total_predictions = 0
        correct_predictions = 0

        for text, text_pred in zip(texts, texts_pred):
            for sentence, sentence_pred in zip(text, text_pred):
                for y in sentence.opinions:
                    for y_pred in sentence_pred.opinions:
                        if (y.nodes == y_pred.nodes) and (y.category == y_pred.category):
                            correct_predictions += 1
                            break
                total_targets += len(sentence.opinions)
                total_predictions += len(sentence_pred.opinions)

        if total_predictions == 0:
            return Score(precision=1.0, recall=0.0, f1=0.0)
        if correct_predictions == 0:
            return Score(precision=0.0, recall=0.0, f1=0.0)
        precision = correct_predictions / total_targets
        recall = correct_predictions / total_predictions
        f1 = 2 * (precision * recall) / (precision + recall)
        return Score(precision=precision, recall=recall, f1=f1)
