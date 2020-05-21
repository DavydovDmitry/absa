from typing import List, Union
from enum import Enum


class Polarity(Enum):
    negative = 0
    neutral = 1
    positive = 2

    @classmethod
    def is_polarity_name(cls, polarity: str):
        if polarity in cls.__members__:
            return True
        return False

    @classmethod
    def get_polarity(cls, polarity: Union['Polarity', str, int]):
        if isinstance(polarity, Polarity):
            return polarity

        # polarity name
        if isinstance(polarity, str):
            if cls.is_polarity_name(polarity):
                return Polarity[polarity]

        # polarity value
        elif isinstance(polarity, int):
            match = [x for x in cls if x.value == polarity]
            if match:
                return match[0]
        return DEFAULT_POLARITY


DEFAULT_POLARITY = Polarity.neutral


class Target:
    """
    Represent target as list of indices. Each index is position in text of
    sentence.

    Parameters:
    ----------
    nodes : list
        Indices of tokens (aspect terms).
    category : str
        Aspect category of target.
    polarity : str
        Polarity of target.
    """
    def __init__(self, nodes: List[int], category: str, polarity: str = ...):
        self.nodes = nodes
        self.category = category
        self.polarity = Polarity.get_polarity(polarity)

    def __str__(self):
        return f'{self.nodes} {self.category} {self.polarity.name}'

    def __hash__(self):
        return hash((tuple(self.nodes), self.category))

    def set_polarity(self, polarity: Union[str, int]):
        self.polarity = Polarity.get_polarity(polarity)

    def reset_polarity(self):
        self.polarity = DEFAULT_POLARITY
