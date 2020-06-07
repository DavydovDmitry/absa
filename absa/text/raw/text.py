"""Module for raw reviews representation
"""

from typing import List

from ..opinion.mixin import OpinionMixin
from .opinion import Opinion


class Text(OpinionMixin):
    def __init__(self, text: str, opinions: List[Opinion] = None):
        if opinions is None:
            opinions = []
        super().__init__(opinions)
        self.text = text

    def display(self) -> None:
        """Color print of review.

        todo
        """

        pass

    def get_text(self) -> str:
        return self.text
