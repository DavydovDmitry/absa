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
    def __init__(self, start_index: int, stop_index: int, category: str, polarity: str = None):
        super().__init__(category, polarity)
        self.start_index = start_index
        self.stop_index = stop_index

    def __str__(self):
        # todo
        return f'{self.start_index} {self.category} {self.polarity.name}'

    def __hash__(self):
        # todo
        return hash(tuple(self.start_index, self.stop_index, self.category))

    def __eq__(self, other: 'Opinion'):
        if not isinstance(other, Opinion):
            return False
        return (self.start_index == other.start_index) & \
               (self.stop_index == other.stop_index) & \
               (self.category == other.category) & \
               (self.polarity == other.polarity)
