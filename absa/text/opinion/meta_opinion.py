from typing import Union

from .polarity import Polarity, DEFAULT_POLARITY


class MetaOpinion:
    """Opinion representation

    Attributes
    ----------
    category : str
        Aspect category of target.
    polarity : Polarity
        Polarity of target.
    """
    def __init__(self, category: str, polarity: str = None):
        self.category = category
        self.polarity = Polarity.get_polarity(polarity)

    def set_polarity(self, polarity: Union[str, int]) -> None:
        self.polarity = Polarity.get_polarity(polarity)

    def reset_polarity(self) -> None:
        self.polarity = DEFAULT_POLARITY
