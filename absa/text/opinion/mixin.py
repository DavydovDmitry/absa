from typing import List, Any, Type

from .meta_opinion import MetaOpinion


class OpinionMixin:
    def __init__(self, opinions: List[MetaOpinion]):
        self.opinions = opinions

    def reset_opinions(self):
        self.opinions = []
