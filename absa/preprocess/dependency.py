from typing import List
import re
import logging
import sys

import networkx as nx
import stanza
from tqdm import tqdm

from absa import PROGRESSBAR_COLUMNS_NUM
from absa.text.raw.text import Text
from absa.text.parsed.opinion import Opinion
from absa.text.parsed.sentence import ParsedSentence
from absa.text.parsed.text import ParsedText

WORD_REG = re.compile(r'(?:\w+-\w+)|(?:\w+)')


def dep_parse_reviews(texts: List[Text], nlp: stanza.Pipeline) -> List[ParsedText]:
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
    texts : List[Text]
        list of raw texts
    nlp : stanza.Pipeline
        pipeline for text processing

    Returns
    ----------
    parsed_texts : List[ParsedText]
        List of parsed reviews
    """

    logging.info('Start dependency parsing...')
    parsed_texts = []

    with tqdm(total=len(texts), ncols=PROGRESSBAR_COLUMNS_NUM,
              file=sys.stdout) as progress_bar:
        for text_index, text in enumerate(texts):
            parsed_sentences = []
            if len(text.get_text()) > 0:
                prev_sentences_len = 0
                for doc in nlp(text.get_text()).sentences:
                    stop_index = prev_sentences_len
                    id2word = dict()
                    id2lemma = dict()
                    id2dep = dict()
                    graph = nx.classes.DiGraph()
                    for token in doc.tokens:
                        token_index = int(token.id[0])
                        word = token.words[0]
                        start_index = text.get_text().find(word.text, stop_index)
                        stop_index = start_index + len(word.text)
                        id2word[token_index] = (start_index - prev_sentences_len,
                                                stop_index - prev_sentences_len)
                        if word.upos in [
                                'PUNCT',
                        ]:
                            continue
                        graph.add_node(token_index)
                        id2lemma[token_index] = word.lemma + '_' + word.upos
                        id2dep[token_index] = word.deprel

                    # Create dependency graph
                    for token in doc.tokens:
                        word = token.words[0]
                        token_index = int(token.id[0])
                        governor = word.head
                        if (token_index in graph) and (governor in graph):
                            graph.add_edge(token_index, governor)

                    # Create opinions
                    parsed_sentence = ParsedSentence(
                        graph=graph,
                        text=text.get_text()[prev_sentences_len:stop_index],
                        id2word=id2word,
                        id2lemma=id2lemma,
                        id2dep=id2dep)

                    if text.opinions:
                        opinions = []
                        for opinion in text.opinions:
                            if (prev_sentences_len <= opinion.start_index) and (
                                    opinion.stop_index <= stop_index):
                                # implicit
                                if opinion.start_index == opinion.stop_index:
                                    opinions.append(
                                        Opinion(nodes=[],
                                                category=opinion.category,
                                                polarity=opinion.polarity))
                                # explicit
                                else:
                                    opinions.append(
                                        Opinion(nodes=parsed_sentence.get_nodes(
                                            start_index=opinion.start_index -
                                            prev_sentences_len,
                                            stop_index=opinion.stop_index -
                                            prev_sentences_len),
                                                category=opinion.category,
                                                polarity=opinion.polarity))
                        parsed_sentence.opinions = opinions
                    parsed_sentences.append(parsed_sentence)
                    prev_sentences_len += len(parsed_sentence.text)
            parsed_texts.append(ParsedText(sentences=parsed_sentences))
            progress_bar.update(1)
    logging.info('\nDependency parsing is completed.')
    return parsed_texts
