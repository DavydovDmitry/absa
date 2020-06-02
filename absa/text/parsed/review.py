from typing import List

from .sentence import ParsedSentence


class ParsedReview:
    def __init__(self, sentences: List[ParsedSentence]):
        self.sentences = sentences

    def __len__(self) -> int:
        return len(self.sentences)

    def __iter__(self) -> ParsedSentence:
        for sentence in self.sentences:
            yield sentence

    def reset_opinions(self):
        for sentence in self:
            sentence.reset_opinions()
