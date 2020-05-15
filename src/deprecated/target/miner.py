from typing import List, Tuple
from functools import reduce
from itertools import chain
import logging
import datetime
import pickle
import sys
import copy

import numpy as np
import pandas as pd
import networkx as nx
from gensim.models import KeyedVectors
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from src import sb12_classifier_path, sb12_train_data_path, PROGRESSBAR_COLUMNS_NUM
from src.review.review import Target
from src.review.parsed_sentence import ParsedSentence
from .aspect_simularity import AttributeMarker, EntityMarker
from .labels import POS_LABELS, DEP_LABELS, ASPECT_LABELS, NONE_ASPECT_LABEL
from .score.metrics import get_sb12


class TargetMiner:
    def __init__(self,
                 word2vec: KeyedVectors,
                 classifier=None,
                 pos_labels=POS_LABELS,
                 dep_labels=DEP_LABELS,
                 aspect_labels=ASPECT_LABELS,
                 none_aspect_label=NONE_ASPECT_LABEL):
        self.classifier = classifier

        self.entity_marker = EntityMarker(word2vec=word2vec)
        self.attribute_marker = AttributeMarker(word2vec=word2vec)
        self.pos_ohe = OneHotEncoder()
        self.pos_ohe.fit_transform(pos_labels)

        self.dep_ohe = OneHotEncoder()
        self.dep_ohe.fit_transform(dep_labels)

        self.none_aspect_label = none_aspect_label
        self.aspect_ohe = OneHotEncoder()
        self.aspect_ohe.fit_transform(aspect_labels)

        self.aspect_le = LabelEncoder()
        self.aspect_le.fit_transform(aspect_labels.ravel())

    @staticmethod
    def _is_target(node_index: int, sentence: ParsedSentence):
        for target in sentence.targets:
            if node_index in target.nodes:
                return 1
        return 0

    def _get_target(self, node_index: int, sentence: ParsedSentence):
        for target in sentence.targets:
            if node_index in target.nodes:
                return target.category
        return self.none_aspect_label

    def _get_columns(self) -> List[str]:
        columns_dep_child = [x + '_child' for x in self.dep_ohe.get_feature_names()]
        columns_pos_child = [x + '_child' for x in self.pos_ohe.get_feature_names()]

        # parent
        columns_dep_parent = [x + '_parent' for x in self.dep_ohe.get_feature_names()]
        columns_pos_parent = [x + '_parent' for x in self.pos_ohe.get_feature_names()]
        columns_aspect_parent = [x + '_parent' for x in self.aspect_ohe.get_feature_names()]

        columns = reduce(
            lambda cs, c: cs + c,
            [
                [
                    'sentence_index',
                    'node_index',
                ],
                columns_dep_parent,
                columns_pos_parent,
                columns_aspect_parent,
                # child
                columns_dep_child,
                columns_pos_child,
                [x + '_dist' for x in self.entity_marker.aspects],
                [x + '_dist' for x in self.attribute_marker.aspects],
            ])
        return columns

    def _get_df_row(self,
                    sentence: ParsedSentence,
                    sentence_index: int,
                    node_child: int,
                    node_parent=None) -> Tuple[np.array, int]:
        if sentence.id2lemma[node_child] not in self.entity_marker.word2vec:
            raise KeyError

        if node_parent is None:
            parent_row = np.concatenate([
                self.dep_ohe.transform([['root']]).toarray(),
                self.pos_ohe.transform([['NOUN']]).toarray(),
                self.aspect_ohe.transform([[NONE_ASPECT_LABEL]]).toarray(),
            ],
                                        axis=1)
        else:
            parent_row = np.concatenate([
                self.dep_ohe.transform([
                    [sentence.id2dep[node_parent].split(':')[0]],
                ]).toarray(),
                self.pos_ohe.transform([
                    [sentence.id2lemma[node_parent].split('_')[1]],
                ]).toarray(),
                self.aspect_ohe.transform(
                    np.array([self._get_target(node_parent, sentence)]).reshape(1,
                                                                                -1)).toarray(),
            ],
                                        axis=1)

        row = np.concatenate(
            [
                np.array([
                    sentence_index,
                    node_child,
                ], dtype=np.int32).reshape(1, -1),
                parent_row,
                # child
                self.dep_ohe.transform([
                    [sentence.id2dep[node_child].split(':')[0]],
                ]).toarray(),
                self.pos_ohe.transform([
                    [sentence.id2lemma[node_child].split('_')[1]],
                ]).toarray(),
                np.array([
                    x for x in self.entity_marker.aspects_dist(
                        sentence.id2lemma[node_child]).values()
                ]).reshape(1, -1),
                np.array([
                    x for x in self.attribute_marker.aspects_dist(
                        sentence.id2lemma[node_child]).values()
                ]).reshape(1, -1),
            ],
            axis=1)

        y = self.aspect_le.transform(
            np.array([
                self._get_target(node_index=node_child, sentence=sentence),
            ]).ravel())
        return row, y

    def get_df(self, sentences: List[ParsedSentence]) -> Tuple[pd.DataFrame, np.array]:
        df = pd.DataFrame(columns=self._get_columns())
        y = list()

        logging.info(f'{datetime.datetime.now().ctime()}. Start extracting features...')
        with tqdm(total=len(sentences), ncols=PROGRESSBAR_COLUMNS_NUM,
                  file=sys.stdout) as progress_bar:
            for sentence_index, sentence in enumerate(sentences):
                if not sentence.graph.nodes:
                    continue

                source = [n for n, d in sentence.graph.out_degree() if d == 0]
                if not source:
                    continue
                source = source[0]

                for node_parent, node_child in chain([[None, source]],
                                                     nx.bfs_edges(sentence.graph,
                                                                  source=source,
                                                                  reverse=True)):
                    try:
                        row, y_row = self._get_df_row(sentence=sentence,
                                                      sentence_index=sentence_index,
                                                      node_child=node_child,
                                                      node_parent=node_parent)
                    except KeyError:
                        pass
                    else:
                        row = pd.DataFrame(data=row.reshape(1, -1),
                                           columns=self._get_columns())
                        df = df.append(row, ignore_index=True)
                        y.append(y_row)
                progress_bar.update(1)
        logging.info(f'{datetime.datetime.now().ctime()}. Extracting features is complete.')
        y = np.array(y).ravel()
        return df, y

    def fit(self, sentences=None, df=None, y=None) -> Tuple[pd.DataFrame, np.array]:
        if self.classifier is None:
            raise AttributeError('Classifier not specified...')

        if (df is None) and (y is None):
            df, y = self.get_df(sentences=sentences)
        self.classifier.fit(df, y)
        return df, y

    def predict(self, sentences: List[ParsedSentence]) -> List[ParsedSentence]:
        sentences = copy.deepcopy(sentences)
        logging.info(f'{datetime.datetime.now().ctime()}. Start making predictions.')
        with tqdm(total=len(sentences), ncols=PROGRESSBAR_COLUMNS_NUM,
                  file=sys.stdout) as progress_bar:
            for sentence_index, sentence in enumerate(sentences):
                if not sentence.graph.nodes:
                    continue

                targets = []
                source = [n for n, d in sentence.graph.out_degree() if d == 0]
                if not source:
                    continue
                source = source[0]

                for node_parent, node_child in chain([[None, source]],
                                                     nx.bfs_edges(sentence.graph,
                                                                  source=source,
                                                                  reverse=True)):
                    try:
                        row, y_row = self._get_df_row(sentence=sentences[sentence_index],
                                                      sentence_index=sentence_index,
                                                      node_child=node_child,
                                                      node_parent=node_parent)
                    except KeyError:
                        pass
                    else:
                        row = pd.DataFrame(data=row.reshape(1, -1),
                                           columns=self._get_columns())

                        y = self.classifier.predict(row)
                        y = self.aspect_le.inverse_transform(y)[0]
                        if y != NONE_ASPECT_LABEL:
                            for target_index, target in enumerate(targets):
                                if (node_parent in target.nodes) and (target.category == y):
                                    targets[target_index].nodes = sorted(
                                        targets[target_index].nodes + [node_child])
                            else:
                                targets.append(Target(nodes=[
                                    node_child,
                                ], category=y))
                        sentences[sentence_index].targets = targets
                progress_bar.update(1)
        logging.info(f'{datetime.datetime.now().ctime()}. Predicting is complete.')
        return sentences

    def score(self, sentences: List[ParsedSentence]):
        sentences_pred = self.predict(sentences)
        return get_sb12(sentences=sentences, sentences_pred=sentences_pred)

    @staticmethod
    def dump_data(df: pd.DataFrame, y: np.array, pathway=sb12_train_data_path):
        with open(pathway, 'wb') as f:
            pickle.dump((df, y), f)

    @staticmethod
    def load_data(pathway=sb12_train_data_path) -> Tuple[pd.DataFrame, np.array]:
        with open(pathway, 'rb') as f:
            df, y = pickle.load(f)
        return df, y

    def dump_classifier(self):
        with open(sb12_classifier_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_classifier() -> 'TargetMiner':
        with open(sb12_classifier_path, 'rb') as f:
            classifier = pickle.load(f)
        return classifier
