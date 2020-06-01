from dataclasses import dataclass


@dataclass
class Score:
    precision: float
    recall: float
    f1: float
