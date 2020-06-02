from typing import List, Union

from .polarity import Polarity, DEFAULT_POLARITY


class Opinion:
    """Opinion representation

    Represent aspect term as list of indices. Each index is position in text of
    sentence.

    Attributes
    ----------
    nodes : list
        Indices of tokens (aspect terms).
    category : str
        Aspect category of target.
    polarity : Polarity
        Polarity of target.
    """
    def __init__(self, nodes: List[int], category: str, polarity: str = None):
        self.nodes = nodes
        self.category = category
        self.polarity = Polarity.get_polarity(polarity)

    def __str__(self):
        return f'{self.nodes} {self.category} {self.polarity.name}'

    def __hash__(self):
        return hash((tuple(self.nodes), self.category))

    def __eq__(self, other: 'Opinion'):
        if not isinstance(other, Opinion):
            return False
        return (self.nodes == other.nodes) & \
               (self.category == other.category) & \
               (self.polarity == other.polarity)

    def set_polarity(self, polarity: Union[str, int]) -> None:
        self.polarity = Polarity.get_polarity(polarity)

    def reset_polarity(self) -> None:
        self.polarity = DEFAULT_POLARITY
