from typing import List, Union, Dict
import logging
import time
import copy

import torch as th
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
from gensim.models import KeyedVectors

from .loader import DataLoader
from .gcn import GCNClassifier
from .utils import torch_utils
from .loader import Batch
from src import SCORE_DECIMAL_LEN
from src.review.parsed_sentence import ParsedSentence
from src import polarity_classifier_dump_path
from .score.display import display_score


class PolarityClassifier:
    """Polarity classifier

    Attributes
    ----------
    word2vec : Union[KeyedVectors, Dict]
        KeyedVectors to set pretrained embeddings.
        Otherwise dict - to set vocabulary and shape of embeddings.
    """
    def __init__(self,
                 word2vec: Union[KeyedVectors, Dict],
                 batch_size=32,
                 emb_matrix_shape=None):
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.batch_size = batch_size

        if isinstance(word2vec, KeyedVectors):
            self.vocabulary = {w: i for i, w in enumerate(word2vec.index2word)}
            emb_matrix = th.FloatTensor(word2vec.vectors)
        elif isinstance(word2vec, dict):
            self.vocabulary = word2vec
            emb_matrix = None
        else:
            raise ValueError

        while True:
            try:
                self.model = GCNClassifier(emb_matrix=emb_matrix,
                                           device=self.device,
                                           emb_matrix_shape=emb_matrix_shape)
            except RuntimeError:
                time.sleep(1)
            else:
                break
        self.model.to(self.device)

    def batch_metrics(self, batch: Batch):
        logits, gcn_outputs = self.model(embed_ids=batch.embed_ids,
                                         adj=batch.adj,
                                         mask=batch.mask,
                                         sentence_len=batch.sentence_len)
        loss = F.cross_entropy(logits, batch.polarity, reduction='mean')
        corrects = (th.max(logits, 1)[1].view(
            batch.polarity.size()).data == batch.polarity.data).sum()
        acc = np.float(corrects) / batch.polarity.size()[0]
        return logits, loss, acc

    def fit(self,
            train_sentences: List[ParsedSentence],
            val_sentences: List[ParsedSentence],
            lr=0.01,
            optimizer_name='adamax',
            num_epoch=5):

        parameters = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch_utils.get_optimizer(optimizer_name, parameters, lr)

        train_batches = DataLoader(sentences=train_sentences,
                                   batch_size=self.batch_size,
                                   vocabulary=self.vocabulary,
                                   device=self.device)
        val_batches = DataLoader(sentences=val_sentences,
                                 batch_size=self.batch_size,
                                 vocabulary=self.vocabulary,
                                 device=self.device)

        train_acc_history, train_loss_history = [], []
        val_acc_history, val_loss_history, f1_score_history = [], [], []

        for epoch in range(1, num_epoch + 1):

            # Train
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
            val_len = len(val_batches)
            self.model.eval()
            predictions, labels = [], []
            val_loss, val_acc = 0., 0.
            for i, batch in enumerate(val_batches):
                logits, loss, acc = self.batch_metrics(batch=batch)
                val_loss += loss.data
                val_acc += acc
                predictions += np.argmax(logits.data.cpu().numpy(), axis=1).tolist()
                labels += batch.polarity.data.cpu().numpy().tolist()
            f1_score = metrics.f1_score(labels, predictions, average='macro')

            train_loss = train_loss / train_len
            train_acc = train_acc / train_len
            val_loss = val_loss / val_len
            val_acc = val_acc / val_len
            logging.info('-' * 40 + f' Epoch {epoch:03d} ' + '-' * 40)
            logging.info(f'Train ' + f'loss: {train_loss:.{SCORE_DECIMAL_LEN}f}| ' +
                         f'acc: {train_acc:.{SCORE_DECIMAL_LEN}f}')
            logging.info(f'Test  ' + f'loss: {(val_loss/val_len):.{SCORE_DECIMAL_LEN}f}| ' +
                         f'acc: {val_acc:.{SCORE_DECIMAL_LEN}f}| ' +
                         f'f1_score: {f1_score:.{SCORE_DECIMAL_LEN}f}')

            train_acc_history.append(train_acc)
            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            f1_score_history.append(f1_score)
        display_score(parameter_values=[x for x in range(num_epoch)],
                      train_values=train_acc_history,
                      val_values=val_acc_history)
        self.save_model()

    def predict(self, sentences: List[ParsedSentence]):
        self.model.eval()
        sentences = copy.deepcopy(sentences)
        batches = DataLoader(sentences=sentences,
                             batch_size=self.batch_size,
                             vocabulary=self.vocabulary,
                             device=self.device)
        for batch_index, batch in enumerate(batches):
            logits, gcn_outputs = self.model(embed_ids=batch.embed_ids,
                                             adj=batch.adj,
                                             mask=batch.mask,
                                             sentence_len=batch.sentence_len)
            pred_labels = np.argmax(logits.data.cpu().numpy(), axis=1).tolist()
            for item_index, item in enumerate(pred_labels):
                sentence_index = batch.sentence_index[item_index]
                target_index = batch.target_index[item_index]
                sentences[sentence_index].targets[target_index].set_polarity(item)
        return sentences

    def save_model(self):
        th.save({
            'vocabulary': self.vocabulary,
            'model': self.model.state_dict()
        }, polarity_classifier_dump_path)

    @staticmethod
    def load_model():
        checkpoint = th.load(polarity_classifier_dump_path)
        model = checkpoint['model']
        classifier = PolarityClassifier(
            word2vec=checkpoint['vocabulary'],
            emb_matrix_shape=[x for x in model['gcn.embed.weight'].shape])
        classifier.model.load_state_dict(model)
        return classifier
