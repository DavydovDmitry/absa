import os

from absa import dumps_path

reviews_dump_path = os.path.join(dumps_path, 'reviews')

from .target import Target
from absa.review.raw.sentence import Sentence
from absa.review.parsed.sentence import ParsedSentence
