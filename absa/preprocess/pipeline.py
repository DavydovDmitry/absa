import os
from typing import Dict, List

from absa import TEST_APPENDIX, train_reviews_path, test_reviews_path, \
    parsed_reviews_dump_path, checked_reviews_dump_path, raw_reviews_dump_path
from absa.text.parsed.review import ParsedReview
from .parse_xml import parse_xml
from .spell_check import spell_check
from .dep_parse import dep_parse_reviews
from ..utils.nlp import NLPPipeline
from ..utils.dump import make_dump, load_dump


def preprocess_pipeline(vocabulary: Dict = None,
                        is_train: bool = True,
                        skip_spell_check: bool = True,
                        make_dumps: bool = True) -> List[ParsedReview]:
    """Pipeline for review preprocess

    Pipeline:
    - reviews parsing - tokenize reviews and create it's representation
    - spell checking
    - building dependency trees

                  |
                  | file_name
                  V
    +------------------------------------- Preprocess pipeline ----------+
    |             |                                                      |
    |             V                                                      |
    |       Upload reviews                                               |
    |             |                                                      |
    |             |     List[Reviews]                                    |
    |             V                                                      |
    |         Spell check                                                |
    |             |                                                      |
    |             |     List[Reviews]                                    |
    |             V                                                      |
    |      Dependency parsing                                            |
    |             |                                                      |
    |             V                                                      |
    +--------------------------------------------------------------------+
                  |
                  |     List[ParsedReview]
                  V

    Make dumps of results of every stage, to not run all stages further.
    """

    # Set postfix for train/test data
    if is_train:
        appendix = ''
        reviews_path = train_reviews_path
    else:
        appendix = TEST_APPENDIX
        reviews_path = test_reviews_path

    # Parse reviews
    if os.path.isfile(raw_reviews_dump_path + appendix):
        reviews = load_dump(pathway=raw_reviews_dump_path + appendix)
    else:
        if vocabulary is None:
            raise ValueError(
                "There's no dump. Need to parse review. Therefore vocabulary parameter is required."
            )
        reviews = parse_xml(vocabulary=vocabulary, pathway=reviews_path)
        if make_dumps:
            make_dump(obj=reviews, pathway=raw_reviews_dump_path + appendix)

    # Spellcheck
    if not skip_spell_check:
        if os.path.isfile(checked_reviews_dump_path + appendix):
            reviews, spell_checked2init = load_dump(pathway=checked_reviews_dump_path +
                                                    appendix)
        else:
            reviews, spell_checked2init = spell_check(reviews)
            if make_dumps:
                make_dump(obj=(reviews, spell_checked2init),
                          pathway=checked_reviews_dump_path + appendix)

    # Dependency parsing
    if os.path.isfile(parsed_reviews_dump_path + appendix):
        parsed_reviews = load_dump(pathway=parsed_reviews_dump_path + appendix)
    else:
        nlp = NLPPipeline.nlp
        parsed_reviews = dep_parse_reviews(reviews, nlp)
        if make_dumps:
            make_dump(obj=parsed_reviews, pathway=parsed_reviews_dump_path + appendix)

    return parsed_reviews
