import os

from absa import train_reviews_path, raw_reviews_dump_path, parsed_reviews_dump_path
from absa.preprocess.parse_xml import parse_xml, load_reviews, dump_reviews
from absa.preprocess.dep_parse import dep_parse_reviews, load_parsed_reviews, dump_parsed_reviews
from absa.utils.nlp import NLPPipeline
from absa.utils.embedding import Embeddings
from . import SemEval2016_filename, test_dumps_path, RAW_POSTFIX, SPELL_POSTFIX, DEP_POSTFIX

# def test_xml_parsing_basic():
#     vocabulary = Embeddings.vocabulary
#     reviews = parse_xml(vocabulary=vocabulary, pathway=SemEval2016_sample)
#     dump_reviews(reviews=reviews, pathway=SemEval2016_sample + RAW_POSTFIX + DUMP_POSTFIX)
#
#
# def test_dep_parsing_basic():
#     nlp = NLPPipeline.nlp
#     reviews = load_reviews(pathway=SemEval2016_sample + RAW_POSTFIX + DUMP_POSTFIX)
#     parsed_reviews = dep_parse_reviews(reviews=reviews, nlp=nlp)
#     dump_parsed_reviews(reviews=parsed_reviews,
#                         pathway=SemEval2016_sample + DEP_POSTFIX + DUMP_POSTFIX)


def test_dep_parsing():
    reviews = load_reviews(pathway=os.path.join(test_dumps_path, SemEval2016_filename +
                                                RAW_POSTFIX))
    parsed_reviews = load_parsed_reviews(
        pathway=os.path.join(test_dumps_path, SemEval2016_filename + DEP_POSTFIX))

    for raw_review, parsed_review in zip(reviews, parsed_reviews):
        for raw_sentence, parsed_sentence in zip(raw_review, parsed_review):
            parsed_sentence = parsed_sentence.to_specified_sentence(text=raw_sentence.text)
            assert raw_sentence == parsed_sentence
