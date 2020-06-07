import os

from absa.utils.embedding import Embeddings
from absa.utils.nlp import NLPPipeline
from absa.utils.dump import make_dump, load_dump
from absa.input.semeval2016 import from_xml
from absa.preprocess.dependency import dep_parse_reviews
from . import SemEval2016_filename, test_dumps_path, RAW_POSTFIX, DEP_POSTFIX, SemEval2016_pathway


def test_xml_parsing_basic():
    vocabulary = Embeddings.vocabulary
    reviews = from_xml(vocabulary=vocabulary, pathway=SemEval2016_pathway)
    make_dump(obj=reviews,
              pathway=os.path.join(test_dumps_path, SemEval2016_filename + RAW_POSTFIX))


def test_dep_parsing_basic():
    nlp = NLPPipeline.nlp
    reviews = load_dump(pathway=os.path.join(test_dumps_path, SemEval2016_filename +
                                             RAW_POSTFIX))
    parsed_reviews = dep_parse_reviews(reviews=reviews, nlp=nlp)
    make_dump(obj=parsed_reviews,
              pathway=os.path.join(test_dumps_path, SemEval2016_filename + DEP_POSTFIX))


def test_dep_parsing():
    reviews = load_dump(pathway=os.path.join(test_dumps_path, SemEval2016_filename +
                                             RAW_POSTFIX))
    parsed_reviews = load_dump(pathway=os.path.join(test_dumps_path, SemEval2016_filename +
                                                    DEP_POSTFIX))

    for raw_review, parsed_review in zip(reviews, parsed_reviews):
        for raw_sentence, parsed_sentence in zip(raw_review, parsed_review):
            if not parsed_sentence.is_opinions_contain_unknown():
                parsed_sentence = parsed_sentence.to_specified_sentence(text=raw_sentence.text)
                assert raw_sentence == parsed_sentence
