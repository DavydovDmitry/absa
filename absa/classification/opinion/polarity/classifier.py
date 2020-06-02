from typing import List, Dict
import logging
import time
import copy
import sys

import torch as th
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
from frozendict import frozendict
from tqdm import tqdm

from absa import SCORE_DECIMAL_LEN, opinion_polarity_classifier_dump_path, PROGRESSBAR_COLUMNS_NUM
from absa.text.parsed.review import ParsedReview
from absa.text.parsed.sentence import ParsedSentence
from absa.text.opinion.opinion import Polarity
from .loader import DataLoader, Batch
from .nn.nn import NeurelNetwork


class PolarityClassifier:
    """Polarity classifier

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
                 batch_size: int = 100):
        # self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.device = th.device('cpu')
        self.batch_size = batch_size

        self.vocabulary = vocabulary
        while True:
            try:
                self.model = NeurelNetwork(emb_matrix=emb_matrix,
                                           device=self.device,
                                           num_class=Polarity.__len__())
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    th.cuda.empty_cache()
            else:
                break

    def batch_metrics(self, batch: Batch):
        """Make a forward step and return metrics"""
        logits = self.model(embed_ids=batch.embed_ids,
                            graph=batch.graph,
                            term_mask=batch.term_mask,
                            sentence_len=batch.sentence_len)
        loss = F.cross_entropy(logits, batch.polarity, reduction='mean')
        corrects = (th.max(logits, 1)[1].view(
            batch.polarity.size()).data == batch.polarity.data).sum()
        acc = np.float(corrects) / batch.polarity.size()[0]
        return logits, loss, acc

    def fit(self,
            train_texts: List[ParsedReview],
            val_texts=None,
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
        optimizer = optimizer_class(parameters, **optimizer_params)

        train_batches = DataLoader(texts=train_texts,
                                   batch_size=self.batch_size,
                                   vocabulary=self.vocabulary,
                                   device=self.device)
        train_acc_history, train_loss_history = [], []

        if val_texts:
            val_batches = DataLoader(texts=val_texts,
                                     batch_size=self.batch_size,
                                     vocabulary=self.vocabulary,
                                     device=self.device)
            val_acc_history, val_loss_history, f1_history = [], [], []

        def epoch_step():
            # Train
            start_time = time.time()
            train_len = len(train_batches)
            self.model.train()
            train_loss, train_acc = 0., 0.
            for i, batch in enumerate(train_batches):
                optimizer.zero_grad()
                _, loss, acc = self.batch_metrics(batch=batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.data
                train_acc += acc

            # Validation
            if val_texts:
                val_len = len(val_batches)
                self.model.eval()
                predictions, labels = [], []
                val_loss, val_acc = 0., 0.
                for i, batch in enumerate(val_batches):
                    logits, loss, acc = self.batch_metrics(batch=batch)
                    val_loss += loss.data
                    val_acc += acc
                    predictions += np.argmax(logits.data.numpy(), axis=1).tolist()
                    labels += batch.polarity.data.numpy().tolist()
                f1_score = metrics.f1_score(labels, predictions, average='macro')

                train_loss = train_loss / train_len
                train_acc = train_acc / train_len
                val_loss = val_loss / val_len
                val_acc = val_acc / val_len
                val_loss_history.append(val_loss)
                val_acc_history.append(val_acc)
                f1_history.append(f1_score)

                if verbose:
                    logging.info('-' * 40 + f' Epoch {epoch:03d} ' + '-' * 40)
                    logging.info(f'Elapsed time: {(time.time() - start_time):.{3}f} sec')
                    logging.info(f'Train      ' +
                                 f'loss: {train_loss:.{SCORE_DECIMAL_LEN}f}| ' +
                                 f'acc: {train_acc:.{SCORE_DECIMAL_LEN}f}')
                    logging.info(f'Validation ' +
                                 f'loss: {(val_loss / val_len):.{SCORE_DECIMAL_LEN}f}| ' +
                                 f'acc: {val_acc:.{SCORE_DECIMAL_LEN}f}| ' +
                                 f'f1_score: {f1_score:.{SCORE_DECIMAL_LEN}f}')
            train_acc_history.append(train_acc)
            train_loss_history.append(train_loss)

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
            return train_acc_history, val_acc_history
        return train_acc_history

    def predict(self, texts: List[ParsedReview]) -> List[ParsedReview]:
        """Modify passed sentences. Define every target polarity.

        Parameters
        ----------
        texts : List[ParsedReview]
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
                             device=self.device)
        for batch_index, batch in enumerate(batches):
            logits = self.model(embed_ids=batch.embed_ids,
                                graph=batch.graph,
                                term_mask=batch.term_mask,
                                sentence_len=batch.sentence_len)
            pred_labels = np.argmax(logits.data.numpy(), axis=1).tolist()
            for item_index, item in enumerate(pred_labels):
                text_index = batch.text_index[item_index]
                sentence_index = batch.sentence_index[item_index]
                opinion_index = batch.opinion_index[item_index]
                texts[text_index].sentences[sentence_index].opinions[
                    opinion_index].set_polarity(item)
        return texts

    def save_model(self):
        """Save model state."""
        th.save({
            'vocabulary': self.vocabulary,
            'model': self.model.state_dict()
        }, opinion_polarity_classifier_dump_path)

    @staticmethod
    def load_model() -> 'PolarityClassifier':
        """Load pretrained model."""
        checkpoint = th.load(opinion_polarity_classifier_dump_path)
        model = checkpoint['model']
        classifier = PolarityClassifier(vocabulary=checkpoint['vocabulary'],
                                        emb_matrix=model['nn.embed.weight'])
        classifier.model.load_state_dict(model)
        return classifier

    @staticmethod
    def score(texts: List[ParsedReview], texts_pred: List[ParsedReview]) -> float:
        """Accuracy of predictions

        Returns
        -------
        accuracy : float
            ratio of correct polarities prediction to total predictions
        """
        correct_predictions = 0
        total_predictions = 0

        for t_index, (t, t_pred) in enumerate(zip(texts, texts_pred)):
            for s_index, (s, s_pred) in enumerate(zip(t, t_pred)):
                if (len(s.opinions) != len(s_pred.opinions)) or \
                   (len(set(hash(t) for t in s.opinions) &
                        set(hash(t) for t in s_pred.opinions)) != len(s.opinions)):
                    logging.error(len(set(s.opinions).intersection(set(s_pred.opinions))))
                    logging.error('-' * 50 + ' Original  targets ' + '-' * 50)
                    for l_target in s.opinions:
                        logging.error(l_target)
                    logging.error('-' * 50 + ' Predicted targets ' + '-' * 50)
                    for l_target_pred in s_pred.opinions:
                        logging.error(l_target_pred)
                    raise ValueError(f'Targets mismatch in {t_index} text {s_index} sentence.')

                for target in s.opinions:
                    for target_pred in s_pred.opinions:
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
