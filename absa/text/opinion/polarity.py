from typing import Union
from enum import IntEnum


class Polarity(IntEnum):
    negative = 0
    neutral = 1
    positive = 2

    @classmethod
    def is_polarity_name(cls, polarity: str) -> bool:
        if polarity in cls.__members__:
            return True
        return False

    @classmethod
    def get_polarity(cls, polarity: Union['Polarity', str, int]) -> 'Polarity':
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

    @classmethod
    def __len__(cls) -> int:
        return len(cls._member_names_)


DEFAULT_POLARITY = Polarity.neutral
