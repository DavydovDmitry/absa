import os
import logging
import sys
from typing import List
from xml.etree import ElementTree

from tqdm import tqdm

from absa.text.raw.text import Text
from absa.text.raw.opinion import Opinion
from absa import PROGRESSBAR_COLUMNS_NUM


def from_xml(pathway: str) -> List[Text]:
    if not os.path.isfile(pathway):
        raise FileNotFoundError(f'There is no such file: {pathway}')

    tree = ElementTree.parse(pathway)
    root = tree.getroot()
    logging.info('Start reviews parsing...')

    reviews = []
    with tqdm(total=len(root), ncols=PROGRESSBAR_COLUMNS_NUM, file=sys.stdout) as progress_bar:
        for review_index, review in enumerate(root):
            text_len = 0
            sentences = []
            opinions = []

            for sentence_index, sentence in enumerate(review.find('sentences')):
                text = sentence.find('text').text

                if sentence.find('Opinions'):
                    for opinion in sentence.find('Opinions'):
                        category = opinion.get('category')
                        polarity = opinion.get('polarity')
                        target_start = int(opinion.get('from')) + text_len
                        target_stop = int(opinion.get('to')) + text_len
                        opinions.append(
                            Opinion(start_index=target_start,
                                    stop_index=target_stop,
                                    category=category,
                                    polarity=polarity))

                sentences.append(text)
                text_len += len(text)
            reviews.append(Text(text=''.join(sentences), opinions=opinions))
            progress_bar.update(1)

    logging.info('Reviews parsing is complete.')
    return reviews
