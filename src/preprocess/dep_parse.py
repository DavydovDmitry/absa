from typing import List
import re
import logging
import pickle
import sys

import networkx as nx
import stanfordnlp
from tqdm import tqdm

from src.review.review import Review, Target
from src.review.parsed_sentence import ParsedSentence

WORD_REG = re.compile(r'(?:\w+-\w+)|(?:\w+)')


def parse_reviews(reviews: List[Review],
                  nlp: stanfordnlp.Pipeline) -> List[List[ParsedSentence]]:
    """Get another representation of reviews.

    Parse reviews sentences to build dependency trees.

    When parse sentence get only words (ignore punctuation). Represent
    sentence as a list of node_id's. All node attribute such as polarity,
    POS, dependency type to parent represent as dicts, where node_id is a
    key.

    I.e. If reviews already have sentiment polarity marks then save its as
    dict key=node_id: int, value=polarity_id: int. By default each node
    have neutral polarity.

    - Order of nodes correspond to order of words in sentence.
    - Sentence can be restore from id2word (every word has id).
      BUT not every word is in id2lemma!!! Filter node by id2lemma!

    Parameters
    ----------
    reviews: list[Review]
    nlp: stanfordnlp.Pipeline

    Returns
    ----------
    List of list parsed sentences
    """

    logging.info('Start dependency parsing...')
    parsed_reviews = []

    with tqdm(total=len(reviews), file=sys.stdout) as progress_bar:
        for review_index, review in enumerate(reviews):
            parsed_sentences = []
            for sentence_index, sentence in enumerate(review.sentences):
                id2word = dict()
                id2lemma = dict()
                id2dep = dict()
                id2prev_id = dict()
                targets = list()
                graph = nx.classes.DiGraph()

                # parse sentence
                if len(sentence.text) > 0:
                    sentence_text = sentence.get_text()
                    docs = nlp(sentence_text).sentences
                    start = 0
                    token_index = -1
                    for doc in docs:
                        total_token_index = token_index + 1
                        for token in doc.tokens:
                            token_index = total_token_index + int(token.index)
                            word = token.words[0]
                            occur_index = sentence_text.find(word.text, start)
                            start = occur_index + len(word.text)
                            total_length = 0
                            for chunk_index, chunk in enumerate(sentence.text):
                                if total_length <= occur_index < total_length + len(chunk):
                                    id2prev_id[token_index] = chunk_index
                                    break
                                else:
                                    total_length += len(chunk)
                            id2word[token_index] = word.text
                            if word.upos == 'PUNCT':
                                continue
                            graph.add_node(token_index)
                            id2lemma[token_index] = word.lemma + '_' + word.upos
                            id2dep[token_index] = word.dependency_relation

                        for token in doc.tokens:
                            token_index = total_token_index + int(token.index)
                            word = token.words[0]
                            if (token_index in graph) and (word.governor in graph):
                                graph.add_edge(token_index, word.governor)

                        if sentence.targets:
                            for target in sentence.targets:
                                parsed_target_nodes = []
                                if target.nodes:
                                    for target_node in target.nodes:
                                        for node_id in [x for x in id2prev_id if x in id2lemma]:
                                            if id2prev_id[node_id] == target_node:
                                                parsed_target_nodes.append(node_id)
                                targets.append(
                                    Target(nodes=parsed_target_nodes,
                                           category=target.category,
                                           polarity=target.polarity))

                parsed_sentences.append(
                    ParsedSentence(graph=graph,
                                   id2word=id2word,
                                   id2lemma=id2lemma,
                                   id2dep=id2dep,
                                   id2prev_id=id2prev_id,
                                   targets=targets))
            parsed_reviews.append(parsed_sentences)
            progress_bar.update(1)
    logging.info('Dependency parsing is complete.')
    return parsed_reviews


def dump_parsed_reviews(parsed_reviews: List[List[ParsedSentence]], file_pathway: str):
    with open(file_pathway, 'wb') as f:
        pickle.dump(parsed_reviews, f)
    logging.info('Make a dump of dependency trees.')


def load_parsed_reviews(file_pathway: str) -> List[List[ParsedSentence]]:
    with open(file_pathway, 'rb') as f:
        parsed_reviews = pickle.load(f)
    logging.info('Upload dependency trees from dump.')
    return parsed_reviews
