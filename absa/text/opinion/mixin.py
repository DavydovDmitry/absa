from typing import List

from .opinion import Opinion


class OpinionMixin:
    def __init__(self, opinions: List[Opinion]):
        self.opinions = opinions

    def reset_opinions(self):
        self.opinions = []
