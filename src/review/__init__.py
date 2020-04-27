import os

from src import dumps_path

reviews_dump_path = os.path.join(dumps_path, 'reviews')

from .target import Target
from .sentence import Sentence
from .review import Review, get_reviews, load_reviews, dump_reviews
from .parsed_sentence import ParsedSentence
