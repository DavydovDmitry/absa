from typing import List
import json
import time
import logging
import sys

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
    text : str
        Text to be splitted
    line_length : int
        Max number of bytes for text chunk

    Return:
    divided_text : List[str]
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
    # todo: logger file handler
    logger = logging.getLogger()
    console_handler = logger.handlers[0]
    logger.removeHandler(console_handler)
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
                            start_index = correction['pos']
                            stop_index = start_index + correction['len']
                            correct = correction['s'][0]

                            corrections.append(
                                Correction(start_index=start_index + text_shift,
                                           stop_index=stop_index + text_shift,
                                           correct=correct))
                            logging.info(
                                f'Correction: {text_part[start_index:stop_index]}  -> {correct}'
                            )
                        if corrections:
                            text_shift += text.correct(corrections=corrections)
                text_shift += len(text_part)

            progress_bar.update(1)
    logger.addHandler(console_handler)
    logging.info('Spell checking is complete.')
    return texts
