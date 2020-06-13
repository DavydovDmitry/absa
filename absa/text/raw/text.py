"""Module for raw reviews representation
"""

from typing import List
from dataclasses import dataclass

from termcolor import colored

from .opinion import Opinion


@dataclass
class Correction:
    start_index: int
    stop_index: int
    correct: str


class Text:
    def __init__(self, text: str, opinions: List[Opinion] = None):
        if opinions is None:
            opinions = []
        self.opinions = opinions
        self.text = text

    def get_text(self) -> str:
        return self.text

    def correct(self, corrections: List[Correction]) -> int:
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
        return total_shift

    def reset_opinions(self):
        self.opinions = []

    def display(self) -> None:
        """Color print of texts

        Set color of words according it's polarity.
        """
        def get_color(polarity: str):
            if polarity == 'positive':
                return 'green'
            elif polarity == 'negative':
                return 'red'
            elif polarity == 'neutral':
                return 'yellow'

        start_index = 0
        for opinion in sorted([
                opinion
                for opinion in self.opinions if (opinion.start_index != opinion.stop_index)
        ],
                              key=lambda x: x.start_index):
            # display only first occurrence
            if opinion.start_index > start_index:
                print(colored(text=self.text[start_index:opinion.start_index]), end='')
                print(colored(text=self.text[opinion.start_index:opinion.stop_index],
                              color=get_color(opinion.polarity.name),
                              attrs=['blink']),
                      end='')
                start_index = opinion.stop_index
        print(colored(text=self.text[start_index:]))
