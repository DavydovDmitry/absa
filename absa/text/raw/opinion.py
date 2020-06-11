from ..opinion.meta_opinion import MetaOpinion


class Opinion(MetaOpinion):
    """Opinion representation for raw texts

    Attributes
    ----------
    start_index : int
        Position of first letter of opinion
    stop_index : int
        Position of last letter (-1)
    """
    def __init__(self, start_index: int, stop_index: int, category: str, polarity: str = None):
        super().__init__(category, polarity)
        self.start_index = start_index
        self.stop_index = stop_index

    def __str__(self):
        return f'slice=({self.start_index}, {self.stop_index}), ' + \
               f'category={self.category}, ' + \
               f'polarity={self.polarity.name}'

    def __hash__(self):
        return hash((self.start_index, self.stop_index, self.category))

    def __eq__(self, other: 'Opinion'):
        if not isinstance(other, Opinion):
            return False
        return (self.start_index == other.start_index) & \
               (self.stop_index == other.stop_index) & \
               (self.category == other.category) & \
               (self.polarity == other.polarity)

    def is_implicit(self):
        if self.start_index == self.stop_index:
            return True
        return False
