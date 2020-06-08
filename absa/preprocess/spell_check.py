from typing import List
import json
import time
import logging
import sys
import re

import requests
from tqdm import tqdm

from absa import PROGRESSBAR_COLUMNS_NUM
from absa.text.raw.text import Text, Correction


def split_text(text: str, line_length=300) -> List[str]:
    """Split text to satisfy requirement on length of text in query.

    There is expecting cyrillic letters. In utf-8 format cyrillic letters
    borrow 2 bytes. Therefore length of encoded text expecting as x2 of
    len(str).

    Parameters:

    Return:
    todo
    """
    line_length = line_length // 2  # cyrillic letters are expecting

    if len(text) <= line_length:
        return [
            text,
        ]

    divided_text = []
    text_chunk = []
    for token in text.split(' '):
        if len(token) + sum([len(t) for t in text_chunk]) - 1 >= line_length:
            divided_text.append(' '.join(text_chunk) + ' ')
            text_chunk = [
                token,
            ]
        else:
            text_chunk.append(token)
    divided_text.append(' '.join(text_chunk))
    return divided_text


def node_preprocess(node: str) -> str:
    node = node.lower()
    for symbol in r'?!.-':
        pattern = r'[' + symbol + r']+'
        node = re.sub(pattern=pattern, repl=symbol, string=node)
    return node


def spell_check(texts: List[Text]) -> List[Text]:
    """Return reviews with correct spelling.

    Change as text as start and stop indexes of terms in opinion.

    Using yandex Speller API. By the way yandex has a limit for requests params
    take a look: https://yandex.ru/dev/speller/doc/dg/concepts/api-overview-docpage/

    WARNING: limit for sentence is unstable.
    Also there is no guarantee that request will processed.
    """

    url = 'https://speller.yandex.net/services/spellservice.json/checkText'

    logging.info('Start spell checking...')
    with tqdm(total=len(texts), ncols=PROGRESSBAR_COLUMNS_NUM,
              file=sys.stdout) as progress_bar:
        for text in texts:
            text_shift = 0
            for text_part in split_text(text.get_text()):

                # Prepare request
                payload = {
                    'text': text_part.encode('utf-8'),
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

                # Handle response
                if res.status_code == 200:
                    if res.content:
                        corrections = []
                        for correction in json.loads(res.content.decode('utf-8')):
                            corrections.append(
                                Correction(start_index=correction['pos'] + text_shift,
                                           stop_index=correction['pos'] + correction['len'] +
                                           text_shift,
                                           correct=correction['s'][0]))
                        if corrections:
                            text_shift += len(text_part) + text.correct(
                                corrections=corrections)

            progress_bar.update(1)
    logging.info('Spell checking is complete.')
    return texts
