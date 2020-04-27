import time
import os
from typing import List
import logging
import datetime
from functools import reduce

import torch as th
import torch.nn.functional as F
from torch.utils.data import DataLoader
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split

from src.review.parsed_sentence import ParsedSentence
from src.review.target import Polarity
from . import tree_lstm_dump_path
from .tree_lstm import TreeLSTM
from .dataset import ABSADataset
# from .batch import collate_batch
from .score.display import display_score

ACC_DECIMAL_LEN = 8


class PolarityClassifier:
    """Target classifier based on LSTM networks.

    This classifier make predictions of sentiment polarity on dependence tree.
    """
    def __init__(
        self,
        word2vec: KeyedVectors,
        num_classes=len(Polarity),
        # architecture params
        bi_lstm_hidden_size=10,
        tree_lstm_hidden_size=20,
        num_layers=1,
        dropout=0.5,
        # batch params
        train_batch_size=100,
        test_batch_size=100,
        # train parameters
        val_ratio=0.3,
        learning_rate=0.5,
        epochs_number=10,
        weight_decay=0.0001,
    ):
        self.word2vec = word2vec

        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.val_ratio = val_ratio
        self.lr = learning_rate
        self.epochs_number = epochs_number
        self.weight_decay = weight_decay

        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')

        while True:
            try:
                self.classifier = TreeLSTM(embeddings=self._get_embeddings(),
                                           bi_lstm_hidden_size=bi_lstm_hidden_size,
                                           tree_lstm_hidden_size=bi_lstm_hidden_size,
                                           num_layers=num_layers,
                                           num_classes=num_classes,
                                           dropout=dropout,
                                           device=self.device).to(self.device)
            except RuntimeError:
                time.sleep(1)
            else:
                break

    def _get_embeddings(self) -> th.Tensor:
        embeddings = th.FloatTensor(self.word2vec.vectors)
        return th.cat((th.zeros(1, embeddings.shape[1], dtype=embeddings.dtype), embeddings),
                      dim=0)

    def fit(self, sentences: List[ParsedSentence], test_sentences, save_state=True):

        # train, val = train_test_split(sentences, test_size=self.val_ratio)

        train_dataset = ABSADataset(sentences=sentences, word2vec=self.word2vec)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=self.train_batch_size,
                                  collate_fn=train_dataset.collate_batch(self.device),
                                  shuffle=True,
                                  num_workers=0)

        val_dataset = ABSADataset(sentences=test_sentences, word2vec=self.word2vec)
        val_loader = DataLoader(dataset=val_dataset,
                                batch_size=self.train_batch_size,
                                collate_fn=val_dataset.collate_batch(self.device),
                                shuffle=False,
                                num_workers=0)

        params = [x for x in self.classifier.parameters() if x.requires_grad]

        total_parameters_number = sum([reduce(lambda x, y: x * y, p.shape) for p in params])
        logging.info(f'Total parameters number: {total_parameters_number}')

        # for p in params:
        #     if p.dim() > 1:
        #         th.nn.init.xavier_uniform_(p)

        # Move model to gpu via calling .cuda() before optimizer specification.
        # Otherwise it will be different objects.
        optimizer = th.optim.Adagrad([{
            'params': params,
            'lr': self.lr,
            'weight_decay': self.weight_decay
        }])

        train_accuracies = []
        val_accuracies = []
        for epoch in range(self.epochs_number):
            train_acc = []
            t_epoch = time.time()

            # Train
            self.classifier.train()
            for step, batch in enumerate(train_loader):
                optimizer.zero_grad()
                logits = self.classifier(batch=batch)

                # predicted class
                pred = th.argmax(logits, 1)
                train_acc.append(th.eq(batch.polarity, pred))

                # mistake
                logp = F.log_softmax(input=logits, dim=1)
                loss = F.nll_loss(logp, batch.polarity, reduction='sum')
                loss.backward()
                optimizer.step()

            train_acc = th.cat(train_acc)
            train_acc = train_acc.sum().cpu().float() / train_acc.shape[0]
            # Validation
            self.classifier.eval()
            val_acc = []
            for step, batch in enumerate(val_loader):
                logits = self.classifier(batch=batch)
                pred = th.argmax(logits, 1)
                val_acc.append(th.eq(batch.polarity, pred))
            val_acc = th.cat(val_acc)
            val_acc = val_acc.sum().cpu().float() / val_acc.shape[0]
            logging.info(f'| Epoch {epoch:05d}' +
                         f'| Train accuracy: {train_acc:.{ACC_DECIMAL_LEN}f}' +
                         f'| Validation accuracy: {val_acc:.{ACC_DECIMAL_LEN}f}' +
                         f'| Time {(time.time() - t_epoch):.4f} sec')
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
        display_score(parameter_values=[x for x in range(self.epochs_number)],
                      train_acc=train_accuracies,
                      val_acc=val_accuracies)

        # if save_state:
        #     if os.path.isfile(tree_lstm_dump_path):
        #         os.remove(tree_lstm_dump_path)
        #     th.save(self.classifier.state_dict(), tree_lstm_dump_path)
        # logging.info(f'{datetime.datetime.now().ctime()}. Save classifier weights.')

    # def load_trained_state(self):
    #     if not os.path.isfile(tree_lstm_dump_path):
    #         raise FileNotFoundError(f'There is no such file: {tree_lstm_dump_path}' +
    #                                 'Train model before making classification...')
    #
    #     self.classifier.load_state_dict(th.load(tree_lstm_dump_path))

    def predict(self, sentences: List[ParsedSentence]) -> List[ParsedSentence]:
        """Classify targets in sentences.

        Upload already trained model and make predictions.
        """

        self.classifier.eval()
        dataset = ABSADataset(sentences=sentences, word2vec=self.word2vec)
        loader = DataLoader(dataset=dataset,
                            batch_size=self.test_batch_size,
                            collate_fn=dataset.collate_batch(self.device),
                            shuffle=False,
                            num_workers=0)

        for step, batch in enumerate(loader):
            graph = batch.graph
            number_of_nodes = graph.number_of_nodes()
            with th.no_grad():
                logits = self.classifier(batch=batch)
            pred = th.argmax(logits, 1)

            pred_index = 0
            for sentence_index, target_index in zip(batch.sentence_index, batch.target_index):
                sentences[sentence_index].targets[target_index].set_polarity(
                    int(pred[pred_index]))
                pred_index += 1

        return sentences
