from typing import List

from ..opinion.meta_opinion import MetaOpinion


class Opinion(MetaOpinion):
    """Opinion representation

    todo

    Attributes
    ----------
    category : str
        Aspect category of target.
    polarity : Polarity
        Polarity of target.
    """
    def __init__(self, nodes: List[int], category: str, polarity: str = None):
        super().__init__(category, polarity)
        self.nodes = nodes

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
