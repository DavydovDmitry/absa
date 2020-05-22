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
