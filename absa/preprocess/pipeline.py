from typing import List
import pathlib
import logging

from absa.utils.dump import make_dump, load_dump, dump_is_exist
from absa.text.raw.text import Text
from absa.text.parsed.text import ParsedText
from absa.io.input.text import from_txt
from absa.io.input.semeval2016 import from_xml
from .spell_check import spell_check
from .dependency import dep_parse_reviews
from ..utils.nlp import NLPPipeline


def _parse_file(pathway: pathlib.Path):
    suffix = pathway.suffix

    logging.info(f'Start parse file {pathway} ...')
    if suffix == '.txt':
        texts = from_txt(pathway)
    elif suffix == '.xml':
        texts = from_xml(pathway)
    else:
        raise NotImplemented
    logging.info(f'Parsing is completed.')
    return texts


def preprocess_file(pathway: pathlib.Path, using_dump: bool = False) -> List[ParsedText]:
    """Preprocess file

    - parse file
    - pass texts to `preprocess_texts`

    Parameters
    ----------
    pathway:
        path to file
    using_dump:

    Return
    ------
    """

    dump_name = pathway.stem
    if using_dump:
        if dump_is_exist(dump_name):
            texts = load_dump(dump_name)
            logging.info(f'Restore texts from dump')
        else:
            texts = _parse_file(pathway)
            make_dump(texts, dump_name)
    else:
        texts = _parse_file(pathway)
    return preprocess_texts(texts=texts, dump_name=dump_name)


def preprocess_texts(texts: List[Text], dump_name: str = None) -> List[ParsedText]:
    """Pipeline for text preprocess

    Pipeline:
    - spell checking
    - building dependency trees

                  |
                  |    List[Text]
                  |
    +------------------------------------- Preprocess pipeline ----------+
    |             |                                                      |
    |             V                                                      |
    |         Spell check                                                |
    |             |                                                      |
    |             |     List[Text]                                       |
    |             V                                                      |
    |      Dependency parsing                                            |
    |             |                                                      |
    |             V                                                      |
    +--------------------------------------------------------------------+
                  |
                  |     List[ParsedText]
                  V
    """
    dump_name = pathlib.Path(dump_name)

    # spell check
    spellcheck_dump = dump_name.with_suffix('.spellcheck')
    if dump_name is not None:
        if dump_is_exist(spellcheck_dump):
            texts = load_dump(spellcheck_dump)
            logging.info('Restore checked texts from dump.')
        else:
            texts = spell_check(texts)
            make_dump(texts, spellcheck_dump)
    else:
        texts = spell_check(texts)

    # dependency parsing
    parsed_dump = dump_name.with_suffix('.parsed')
    if dump_name is not None:
        if dump_is_exist(parsed_dump):
            parsed_texts = load_dump(parsed_dump)
            logging.info('Restore parsed texts from dump.')
        else:
            nlp = NLPPipeline.nlp
            parsed_texts = dep_parse_reviews(texts, nlp)
            make_dump(parsed_texts, parsed_dump)
    else:
        nlp = NLPPipeline.nlp
        parsed_texts = dep_parse_reviews(texts, nlp)

    return parsed_texts
