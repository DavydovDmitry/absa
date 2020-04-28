import os
import xml.etree.ElementTree as ET
import logging
import sys
import time
import warnings

import stanfordnlp
from gensim.models import KeyedVectors

from src import TEST_APPENDIX
from src import train_reviews_path, test_reviews_path, parsed_reviews_dump_path, checked_reviews_dump_path
from src.review import get_reviews, load_reviews, dump_reviews, reviews_dump_path
from .spell_check import spell_check, load_checked_reviews, dump_checked_reviews
from .dep_parse import parse_reviews, load_parsed_reviews, dump_parsed_reviews


def preprocess_pipeline(word2vec: KeyedVectors = ..., is_train=True):
    """Pipeline for review's preprocessing

    Pipeline:
    - reviews parsing - tokenize reviews and create it's representation
    - spell checking
    - building dependency trees

    Upload reviews
         |
         |  List[Reviews]
         V
    Spell check
         |
         |  List[Reviews]
         V
    Dependency parsing
         |
         |  List[List[ParsedSentence]]
         V
    Labeling

    Make dumps of results of every stage, to not run all stages further.
    """

    if is_train:
        appendix = ''
        reviews_path = train_reviews_path
    else:
        appendix = TEST_APPENDIX
        reviews_path = test_reviews_path

    # Parse reviews
    if os.path.isfile(reviews_dump_path + appendix):
        reviews = load_reviews(reviews_dump_path + appendix)
    else:
        tree = ET.parse(reviews_path)
        root = tree.getroot()
        if not isinstance(word2vec, KeyedVectors):
            raise ValueError(
                "There's no dump. Need to parse review. Therefore word2vec parameter is required."
            )
        initial_reviews = get_reviews(root, word2vec)
        reviews = [x.get_normalized() for x in initial_reviews]
        dump_reviews(reviews, reviews_dump_path + appendix)

    # Spellcheck
    if os.path.isfile(checked_reviews_dump_path + appendix):
        reviews, spell_checked2init = load_checked_reviews(
            file_pathway=checked_reviews_dump_path + appendix)
    else:
        reviews, spell_checked2init = spell_check(reviews)
        dump_checked_reviews((reviews, spell_checked2init),
                             file_pathway=checked_reviews_dump_path + appendix)

    # Dependency parsing
    if os.path.isfile(parsed_reviews_dump_path + appendix):
        parsed_reviews = load_parsed_reviews(file_pathway=parsed_reviews_dump_path + appendix)
    else:
        logging.info('Prepare for dep parsing...')
        sys.stdout = open(os.devnull, 'w')
        while True:
            try:
                nlp = stanfordnlp.Pipeline(lang='ru', )
            except RuntimeError:
                time.sleep(5)
            else:
                break
        warnings.filterwarnings("ignore", category=UserWarning)
        sys.stdout = sys.__stdout__

        parsed_reviews = parse_reviews(reviews, nlp)
        dump_parsed_reviews(parsed_reviews, file_pathway=parsed_reviews_dump_path + appendix)

    return parsed_reviews
