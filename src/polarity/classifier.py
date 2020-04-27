from typing import List
import logging
import time
import copy

import torch as th
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
from gensim.models import KeyedVectors

from .draw import draw_curve
from .loader import DataLoader
from .gcn import GCNClassifier
from .utils import torch_utils
from .loader import Batch

from src.review.parsed_sentence import ParsedSentence

DECIMAL_LEN = 6


class PolarityClassifier:
    def __init__(self, word2vec: KeyedVectors, batch_size=32):
        self.word2vec = word2vec
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.batch_size = batch_size

        while True:
            try:
                self.model = GCNClassifier(emb_matrix=th.FloatTensor(self.word2vec.vectors),
                                           device=self.device)
            except RuntimeError:
                time.sleep(1)
            else:
                break
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
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

        optimizer = torch_utils.get_optimizer(optimizer_name, self.parameters, lr)

        train_batches = DataLoader(sentences=train_sentences,
                                   batch_size=self.batch_size,
                                   word2vec=self.word2vec,
                                   device=self.device)
        val_batches = DataLoader(sentences=val_sentences,
                                 batch_size=self.batch_size,
                                 word2vec=self.word2vec,
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
            logging.info(f'TRAIN ' + f'loss: {train_loss:.{DECIMAL_LEN}f}| ' +
                         f'acc: {train_acc:.{DECIMAL_LEN}f}')
            logging.info(f'TEST  ' + f'loss: {(val_loss/val_len):.{DECIMAL_LEN}f}| ' +
                         f'acc: {val_acc:.{DECIMAL_LEN}f}| ' +
                         f'f1_score: {f1_score:.{DECIMAL_LEN}f}')

            train_acc_history.append(train_acc)
            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            f1_score_history.append(f1_score)

        bt_train_acc = max(train_acc_history)
        bt_train_loss = min(train_loss_history)
        bt_test_acc = max(val_acc_history)
        bt_f1_score = f1_score_history[val_acc_history.index(bt_test_acc)]
        bt_test_loss = min(val_loss_history)
        # logging.info(
        #     'best train_acc: {},', 'best train_loss: {},', 'best test_acc/f1_score: {}/{}',
        #     'best test_loss: {}'.format(bt_train_acc, bt_train_loss, bt_test_acc, bt_f1_score,
        #                                 bt_test_loss))
        # draw_curve(train_log=train_acc_history,
        #            test_log=val_acc_history[1:],
        #            curve_type="acc",
        #            epoch=num_epoch)
        # draw_curve(train_log=train_loss_history,
        #            test_log=val_loss_history,
        #            curve_type="loss",
        #            epoch=num_epoch)

    def predict(self, sentences: List[ParsedSentence]):
        sentences = copy.deepcopy(sentences)
        batches = DataLoader(sentences=sentences,
                             batch_size=self.batch_size,
                             word2vec=self.word2vec,
                             device=self.device)
        for batch_index, batch in enumerate(batches):
            logits, gcn_outputs = self.model(embed_ids=batch.embed_ids,
                                             adj=batch.adj,
                                             mask=batch.mask,
                                             sentence_len=batch.sentence_len)
            # pred_labels = th.max(logits,1)[1].view(batch.polarity.size()).data.cpu().numpy().tolist()
            pred_labels = np.argmax(logits.data.cpu().numpy(), axis=1).tolist()
            for item_index, item in enumerate(pred_labels):
                sentence_index = batch.sentence_index[item_index]
                target_index = batch.target_index[item_index]
                sentences[sentence_index].targets[target_index].set_polarity(item)
        return sentences
