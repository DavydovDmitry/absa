"""Module for raw reviews representation
"""

from typing import List
from dataclasses import dataclass

from ..opinion.mixin import OpinionMixin
from .opinion import Opinion


@dataclass
class Correction:
    start_index: int
    stop_index: int
    correct: str


class Text(OpinionMixin):
    def __init__(self, text: str, opinions: List[Opinion] = None):
        if opinions is None:
            opinions = []
        super().__init__(opinions)
        self.text = text

    def display(self) -> None:
        """Color print of review.

        todo
        """

        pass

    def get_text(self) -> str:
        return self.text

    def correct(self, corrections: List[Correction]):
        total_shift = 0
        for correction in corrections:
            self.text = self.text[:correction.start_index + total_shift] + \
                        correction.correct + \
                        self.text[correction.stop_index + total_shift:]
            text_shift = len(correction.correct) - \
                         (correction.stop_index - correction.start_index)
            for opinion in self.opinions:
                # todo: typing
                if opinion.start_index > correction.stop_index:
                    opinion.start_index += text_shift
                    opinion.stop_index += text_shift
                elif (opinion.start_index < correction.start_index) and (
                        opinion.stop_index >= correction.stop_index):
                    opinion.stop_index += text_shift
            total_shift += text_shift
