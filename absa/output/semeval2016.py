from typing import List
from xml.etree.ElementTree import Element, SubElement, Comment, tostring
from xml.etree import ElementTree
from xml.dom import minidom
import os

from absa import output_path
from ..text.parsed.text import ParsedText


def to_xml(texts: List[ParsedText], output_filename='parsed.xml') -> None:
    output_file = os.path.join(output_path, output_filename)

    top = Element('Reviews')
    for text in texts:
        review_xml = SubElement(top, 'Review')
        for sentence in text.sentences:
            sentence_xml = SubElement(review_xml, 'Sentence')
            sentence_xml.text = sentence.get_text()
            if sentence.opinions:
                opinions_xml = SubElement(sentence_xml, 'Opinions')
                for opinion in sentence.opinions:
                    target = []
                    for node in opinion.nodes:
                        start, stop = sentence.id2word[node]
                        target.append(sentence.get_text()[start:stop])
                    SubElement(
                        opinions_xml, 'Opinion', {
                            'target': ' '.join(target),
                            'category': opinion.category,
                            'polarity': opinion.polarity.name
                        })
    rough_string = ElementTree.tostring(top, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    with open(output_file, 'w') as f:
        f.write(reparsed.toprettyxml(indent="  "))
