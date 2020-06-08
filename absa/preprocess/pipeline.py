import os
from typing import Dict, List

from absa import TEST_APPENDIX, train_reviews_path, test_reviews_path, \
    parsed_reviews_dump_path, checked_reviews_dump_path, raw_reviews_dump_path
from absa.text.raw.text import Text
from absa.text.parsed.text import ParsedText
from absa.input.semeval2016 import from_xml
from .spell_check import spell_check
from .dependency import dep_parse_reviews
from ..utils.nlp import NLPPipeline
from ..utils.dump import make_dump, load_dump


def preprocess_pipeline(texts: List[Text]) -> List[ParsedText]:
    """Pipeline for review preprocess

    Pipeline:
    - spell checking
    - building dependency trees

                  |    List[Review]
                  |
    +------------------------------------- Preprocess pipeline ----------+
    |             |                                                      |
    |             V                                                      |
    |         Spell check                                                |
    |             |                                                      |
    |             |     List[Review]                                     |
    |             V                                                      |
    |      Dependency parsing                                            |
    |             |                                                      |
    |             V                                                      |
    +--------------------------------------------------------------------+
                  |
                  |     List[ParsedReview]
                  V

    """

    texts = spell_check(texts)
    nlp = NLPPipeline.nlp
    parsed_texts = dep_parse_reviews(texts, nlp)
    return parsed_texts
