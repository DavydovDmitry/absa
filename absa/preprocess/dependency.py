from typing import List
import re
import logging
import sys

import networkx as nx
import stanfordnlp
from tqdm import tqdm

from absa import PROGRESSBAR_COLUMNS_NUM
from absa.text.raw.text import Text
from absa.text.parsed.opinion import Opinion
from absa.text.parsed.sentence import ParsedSentence
from absa.text.parsed.review import ParsedReview

WORD_REG = re.compile(r'(?:\w+-\w+)|(?:\w+)')


def dep_parse_reviews(texts: List[Text], nlp: stanfordnlp.Pipeline) -> List[ParsedReview]:
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
    texts: list[Text]
        list of raw reviews
    nlp: stanfordnlp.Pipeline
        pipeline for text processing

    Returns
    ----------
    parsed_reviews : List[ParsedReview]
        List of parsed reviews
    """

    logging.info('Start dependency parsing...')
    parsed_reviews = []

    with tqdm(total=len(texts), ncols=PROGRESSBAR_COLUMNS_NUM,
              file=sys.stdout) as progress_bar:
        for text_index, text in enumerate(texts):
            parsed_sentences = []
                id2word = dict()
                id2lemma = dict()
                id2dep = dict()
                id2prev_id = dict()
                targets = list()
                graph = nx.classes.DiGraph()

                if len(text.get_text()) > 0:
                    docs = nlp(text.get_text()).sentences
                    start = 0
                    token_index = 0
                    for doc in docs:
                        total_token_index = token_index
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

                        # Create dependency graph from id2lemma nodes
                        for token in doc.tokens:
                            word = token.words[0]
                            token_index = total_token_index + int(token.index)
                            governor = total_token_index + word.governor
                            if (token_index in graph) and (governor in graph):
                                graph.add_edge(token_index, governor)

                        # Create targets with nodes from id2lemma nodes
                        if sentence.opinions:
                            for target in sentence.opinions:
                                parsed_target_nodes = []
                                if target.nodes:
                                    for target_node in target.nodes:
                                        for node_id in [
                                                x for x in id2prev_id if x in id2lemma
                                        ]:
                                            if id2prev_id[node_id] == target_node:
                                                parsed_target_nodes.append(node_id)
                                targets.append(
                                    Opinion(nodes=parsed_target_nodes,
                                            category=target.category,
                                            polarity=target.polarity))

                parsed_sentences.append(
                    ParsedSentence(graph=graph,
                                   id2word=id2word,
                                   id2lemma=id2lemma,
                                   id2dep=id2dep,
                                   id2init_id=id2prev_id,
                                   opinions=targets))
            parsed_reviews.append(ParsedReview(sentences=parsed_sentences))
            progress_bar.update(1)
    logging.info('Dependency parsing is complete.')
    return parsed_reviews
