from typing import List

from termcolor import colored

from ..opinion.opinion import Opinion
from ..opinion.mixin import OpinionMixin


class Sentence(OpinionMixin):
    def __init__(self, text: List[str], opinions: List[Opinion] = None):
        if opinions is None:
            opinions = []
        super().__init__(opinions=opinions)

        self.text = text

    def __repr__(self):
        return f'Text: {self.text}\n' + \
               f'Opinions: {"; ".join(map(lambda x: str(x), self.opinions))}'

    def __eq__(self, other: 'Sentence'):
        if not isinstance(other, Sentence):
            return False
        return (self.text == other.text) & (self.opinions == other.opinions)

    def get_text(self) -> str:
        return ''.join(self.text)

    def get_normalized(self) -> 'Sentence':
        tokens = []
        if self.text:
            for token in self.text:
                tokens.append(token.strip() + ' ')
            tokens[-1] = tokens[-1][:-1]
        return Sentence(text=tokens, opinions=self.opinions)

    def display(self) -> None:
        """Print sentence with according colors."""
        def get_color(polarity: str):
            if polarity == 'positive':
                return 'green'
            elif polarity == 'negative':
                return 'red'
            elif polarity == 'neutral':
                return 'yellow'

        on_color = None
        if self.opinions:
            if not self.opinions[0].nodes:
                on_color = 'on_' + get_color(self.opinions[0].polarity.name)

        for token_index, token in enumerate(self.text):
            for target in self.opinions:
                if token_index in target.nodes:
                    print(colored(text=token,
                                  color=get_color(target.polarity.name),
                                  on_color=on_color,
                                  attrs=['blink']),
                          end='')
                    break
            else:
                print(colored(text=token, on_color=on_color), end='')
