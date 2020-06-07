import os
from typing import List

from absa import input_path
from absa.text.raw.text import Text


def from_txt(filename='raw') -> List[Text]:
    input_file = os.path.join(input_path, filename)

    reviews = []
    with open(input_file, 'rb') as f:
        for text in f.readlines():
            pass
            # todo
