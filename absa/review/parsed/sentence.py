from typing import List, Dict

import networkx as nx

from ..raw.sentence import Sentence
from ..target import Target


class ParsedSentence:
    """
    Attributes
    ----------
    graph ; nx.DiGraph
        dependency tree of sentence
    id2word : dict
        token and it's index (all tokens)
    id2lemma : dict
        node of dependency tree and it's index
    """
    def __init__(self, graph: nx.DiGraph, id2word: Dict[int, str], id2lemma: Dict[int, str],
                 id2dep: Dict[int, str], id2prev_id: Dict[int, int], targets: List[Target]):
        self.graph = graph
        self.id2word = id2word
        self.id2lemma = id2lemma
        self.id2dep = id2dep
        self.targets = targets
        self.id2prev_id = id2prev_id  # backward compatibility

    def get_sentence_order(self) -> List[int]:
        return [
            node_id for node_id, _ in sorted(self.id2prev_id.items(), key=lambda item: item[1])
            if node_id in self.id2lemma
        ]

    def to_sentence(self) -> Sentence:
        """Convert to instance of Sentence class"""
        text = []
        for parsed_node_id, _ in sorted(self.id2prev_id.items(), key=lambda item: item[1]):
            if parsed_node_id in self.id2word:
                text.append(self.id2word[parsed_node_id])

        targets = []
        for target in self.targets:
            target_nodes = []
            for parsed_node_id in target.nodes:
                target_nodes.append(self.id2prev_id[parsed_node_id])
            targets.append(
                Target(nodes=target_nodes, category=target.category, polarity=target.polarity))
        return Sentence(text=text, targets=targets)

    def __len__(self):
        return len(self.graph)

    def reset_targets(self):
        self.targets = []

    def reset_targets_polarities(self):
        for target in self.targets:
            target.reset_polarity()
