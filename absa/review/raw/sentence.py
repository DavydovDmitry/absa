from typing import List

from termcolor import colored

from absa.review.target import Target


class Sentence:
    def __init__(self, text: List[str], targets: List[Target]):
        self.text = text
        self.targets = targets

    def __repr__(self):
        return f'Text: {self.text}\n' + \
               f'Targets: {"; ".join(map(lambda x: str(x), self.targets))}'

    def get_text(self):
        return ''.join(self.text)

    def get_normalized(self) -> 'Sentence':
        tokens = []
        if self.text:
            for token in self.text:
                tokens.append(token.strip() + ' ')
            tokens[-1] = tokens[-1][:-1]
        return Sentence(text=tokens, targets=self.targets)

    def display(self):
        """Print sentence with according colors."""
        def get_color(polarity: str):
            if polarity == 'positive':
                return 'green'
            elif polarity == 'negative':
                return 'red'
            elif polarity == 'neutral':
                return 'yellow'

        on_color = None
        if self.targets:
            if not self.targets[0].nodes:
                on_color = 'on_' + get_color(self.targets[0].polarity.name)

        for token_index, token in enumerate(self.text):
            for target in self.targets:
                if token_index in target.nodes:
                    print(colored(text=token,
                                  color=get_color(target.polarity.name),
                                  on_color=on_color,
                                  attrs=['blink']),
                          end='')
                    break
            else:
                print(colored(text=token, on_color=on_color), end='')
