from typing import List

from .target import Target


class TargetMixin:
    def __init__(self, targets: List[Target]):
        self.targets = targets

    def reset_targets(self):
        self.targets = []
