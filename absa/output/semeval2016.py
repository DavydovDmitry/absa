from typing import List
from xml.etree.ElementTree import Element, SubElement, Comment, tostring
from xml.etree import ElementTree
from xml.dom import minidom
import os

from absa import output_path
from ..text.raw.text import Text


def to_xml(texts: List[Text], output_filename='parsed') -> None:
    output_file = os.path.join(output_path, output_filename)

    top = Element('Reviews')
    for text in texts:
        review_xml = SubElement(top, 'Review')
        for sentence in text.sentences:
            sentence_xml = SubElement(review_xml, 'Sentence')
            sentence_xml.text = sentence.text
            if sentence.opinions:
                opinions_xml = SubElement(sentence_xml, 'Opinions')
                for opinion in sentence.opinions:
                    opinion_xml = SubElement(
                        opinions_xml, 'Opinion', {
                            'target':
                            ' '.join(map(lambda index: sentence.text[index], opinion.nodes)),
                            'category':
                            opinion.category,
                            'polarity':
                            opinion.polarity
                        })
    rough_string = ElementTree.tostring(top, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    with open(output_file, 'w') as f:
        f.write(reparsed.toprettyxml(indent="  "))
