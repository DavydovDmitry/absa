import os
from typing import List

from absa import input_path
from absa.text.raw.text import Text


def from_txt(filename='raw.txt') -> List[Text]:
    input_file = os.path.join(input_path, filename)

    reviews = []
    with open(input_file, 'r') as f:
        for text in f.readlines():
            reviews.append(Text(text=text))
    return reviews
