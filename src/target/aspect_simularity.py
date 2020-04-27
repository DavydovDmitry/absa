import os
from typing import Dict

from gensim.models import KeyedVectors
import numpy as np

from . import attributes_path, entities_path


class AspectMarker:
    def __init__(self, word2vec: KeyedVectors, aspect_path):
        self.word2vec = word2vec
        self.aspects = [
            x for x in os.listdir(aspect_path) if not os.path.isdir(x) and x[0] != '#'
        ]
        self.targets = dict()
        for aspect in self.aspects:
            with open(os.path.join(aspect_path, aspect)) as f:
                self.targets[aspect] = [x.strip() for x in f.readlines() if x]

    def words_dist(self, w1: str, w2: str) -> float:
        return self.word2vec.distance(w1, w2)

    def aspect_dist(self, word: str, aspect: str) -> float:
        if word not in self.word2vec:
            return np.finfo(np.float32).max

        if aspect not in self.aspects:
            raise NameError(f'There is no such aspect: {aspect}.')

        neighbour_number = 5
        closest_dist = [np.inf]
        for target in self.targets[aspect]:
            dist = self.words_dist(word, target)
            for target_index in range(min(len(closest_dist), neighbour_number)):
                if dist < closest_dist[target_index]:
                    closest_dist.insert(target_index, dist)
                    break
        closest_dist = closest_dist[:neighbour_number]
        return sum(closest_dist)

    def aspects_dist(self, word: str) -> Dict:
        dists = dict()
        for aspect in self.aspects:
            dists[aspect] = self.aspect_dist(word=word, aspect=aspect)
        return dists


class EntityMarker(AspectMarker):
    def __init__(self, word2vec: KeyedVectors):
        super().__init__(word2vec, aspect_path=entities_path)


class AttributeMarker(AspectMarker):
    def __init__(self, word2vec: KeyedVectors):
        super().__init__(word2vec, aspect_path=attributes_path)
