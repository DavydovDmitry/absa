"""Module for raw reviews representation
"""

from typing import List

from absa.text.raw.sentence import Sentence


class Review:
    def __init__(self, sentences: List[Sentence]):
        self.sentences = sentences

    def __repr__(self):
        nl = '\n'
        return f'{f"{nl}".join(map(lambda x : str(x), self.sentences))}'

    def __len__(self):
        return len(self.sentences)

    def __iter__(self):
        for sentence in self.sentences:
            yield sentence

    def display(self) -> None:
        """Color print of review."""

        for sentence in self.sentences:
            sentence.display()
            print(end=' ')

    def get_text(self) -> str:
        return ' '.join([x.get_text() for x in self.sentences])

    def get_normalized(self) -> 'Review':
        return Review(sentences=[sentence.get_normalized() for sentence in self.sentences])
