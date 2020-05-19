from typing import List, Dict, Tuple
import json
import pickle
import time
import logging
import sys
from copy import deepcopy
import re

import requests
from tqdm import tqdm

from absa import PROGRESSBAR_COLUMNS_NUM
from absa.review.raw.review import Review


def split_text(text: List[str], line_length=300) -> List[List[str]]:
    """Split text to satisfy requirement on length of text in query.

    There is expecting cyrillic letters. In utf-8 format cyrillic letters
    borrow 2 bytes. Therefore length of encoded text expecting as x2 of
    len(str).
    """
    line_length = line_length // 2  # cyrillic letters are expecting

    if len(''.join(text)) <= line_length:
        return [
            text,
        ]
    divided_text = []
    text_chunk = []
    for token in text:
        if len(token) + sum([len(t) for t in text_chunk]) - 1 >= line_length:
            divided_text.append(text_chunk)
            text_chunk = [
                token,
            ]
        else:
            text_chunk.append(token)
    divided_text.append(text_chunk)
    return divided_text


def node_preprocess(node: str) -> str:
    node = node.lower()
    for symbol in r'?!.-':
        pattern = r'[' + symbol + r']+'
        node = re.sub(pattern=pattern, repl=symbol, string=node)
    return node


def spell_check(reviews: List[Review],
                start_review_index=0) -> Tuple[List[Review], List[List[Dict]]]:
    """Return reviews with correct spelling.

    Using yandex Speller API. By the way yandex has a limit for requests params
    take a look: https://yandex.ru/dev/speller/doc/dg/concepts/api-overview-docpage/

    WARNING: limit for sentence is unstable.
    Also there is no guarantee that request will processed.
    """
    url = 'https://speller.yandex.net/services/spellservice.json/checkText'

    spell_checked2init = [[dict() for _ in review.sentences] for review in reviews]
    logging.info('Start spell checking...')
    with tqdm(total=len(reviews), ncols=PROGRESSBAR_COLUMNS_NUM,
              file=sys.stdout) as progress_bar:
        for review_index in range(start_review_index, len(reviews)):
            review = reviews[review_index]
            for sentence_index, sentence in enumerate(review.sentences):
                sentence_copy = deepcopy(sentence)
                text_parts = list()
                text_shift = 0
                for text_part in split_text(sentence.text):
                    united_part = ''.join(text_part)
                    payload = {
                        'text': united_part.encode('utf-8'),
                        'lang': 'ru',
                        'format': 'plain',
                    }
                    while True:
                        res = requests.post(url=url, data=payload)
                        if res.status_code != 200:
                            if res.status_code >= 500:
                                time.sleep(5)
                                continue
                        break

                    # API works properly
                    if res.status_code == 200:
                        if res.content:
                            corrections = json.loads(res.content.decode('utf-8'))
                            for correction in corrections:
                                token_index = 0
                                start = 0
                                while token_index < len(text_part):
                                    token = text_part[token_index]
                                    token_length = len(token)
                                    if start == correction['pos']:
                                        appendix = sentence.text[
                                            text_shift + token_index][correction['len']:]
                                        # One to many
                                        #      o
                                        #     /|\
                                        #    o o o
                                        if ' ' in correction['s'][0]:
                                            sentence_copy.text[text_shift + token_index] = [
                                                x + ' ' for x in correction['s'][0].split(' ')
                                            ]

                                        # Bijection
                                        #     o
                                        #     |
                                        #     o
                                        elif len(token.strip()) == correction['len']:
                                            sentence_copy.text[
                                                text_shift +
                                                token_index] = correction['s'][0] + appendix

                                        # Many to one
                                        #    o o o
                                        #     \|/
                                        #      o
                                        elif len(token.strip()) < correction['len']:
                                            sentence_copy.text[
                                                text_shift +
                                                token_index] = correction['s'][0] + ' '
                                            while token_length < correction['len']:
                                                token_index += 1
                                                sentence_copy.text[text_shift +
                                                                   token_index] = ''
                                                token_length += len(
                                                    text_part[token_index].strip()) + 1

                                        break
                                    else:
                                        start += token_length
                                    token_index += 1
                    text_shift += len(text_part)
                    text_parts.append(text_part)

                prev_index = 0
                curr_index = 0
                sentence_length = len(sentence_copy.text)
                while prev_index < sentence_length:
                    # Many to one
                    if not sentence_copy.text[curr_index]:
                        for target_index, target in enumerate(sentence_copy.targets):
                            for node_index, node in enumerate(target.nodes):
                                if node > curr_index:
                                    sentence_copy.targets[target_index].nodes[node_index] -= 1
                        sentence_copy.text.pop(curr_index)

                        if curr_index in spell_checked2init[review_index][sentence_index]:
                            spell_checked2init[review_index][sentence_index][
                                curr_index].append(prev_index)
                        else:
                            spell_checked2init[review_index][sentence_index][curr_index] = [
                                prev_index,
                            ]

                    # One to many
                    elif isinstance(sentence_copy.text[curr_index], list):
                        for add_node_index, add_node in enumerate(
                                sentence_copy.text[curr_index]):
                            sentence_copy.text.insert(curr_index + add_node_index + 1,
                                                      add_node)
                            spell_checked2init[review_index][sentence_index][
                                curr_index + add_node_index] = [
                                    prev_index,
                                ]
                            if add_node_index == 0:
                                continue

                            for target_index, target in enumerate(sentence_copy.targets):
                                # for next targets (include current)
                                for node_index, node in enumerate(target.nodes):
                                    if node > curr_index:
                                        sentence_copy.targets[target_index].nodes[
                                            node_index] += 1
                                # for current target
                                if curr_index in target.nodes:
                                    sentence_copy.targets[target_index].nodes.insert(
                                        target.nodes.index(curr_index) + 1,
                                        curr_index + add_node_index)
                        n_nodes = len(sentence_copy.text[curr_index])
                        sentence_copy.text.pop(curr_index)
                        curr_index += n_nodes

                    # Bijection
                    else:
                        if curr_index in spell_checked2init[review_index][sentence_index]:
                            spell_checked2init[review_index][sentence_index][
                                curr_index].append(prev_index)
                        else:
                            spell_checked2init[review_index][sentence_index][curr_index] = [
                                prev_index,
                            ]
                        curr_index += 1
                    prev_index += 1
                sentence_copy.text = [node_preprocess(x) for x in sentence_copy.text]
                reviews[review_index].sentences[sentence_index] = sentence_copy
            progress_bar.update(1)
    logging.info('Spell checking is complete.')
    return reviews, spell_checked2init


def dump_checked_reviews(checked_reviews: Tuple[List[Review], List[List[Dict]]], file_pathway):
    with open(file_pathway, 'wb') as f:
        pickle.dump(checked_reviews, f)
    logging.info('Create a dump of checked reviews.')


def load_checked_reviews(file_pathway) -> Tuple[List[Review], List[List[Dict]]]:
    with open(file_pathway, 'rb') as f:
        checked_reviews = pickle.load(f)
    logging.info('Upload checked reviews from dump.')
    return checked_reviews
