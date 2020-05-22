from typing import List, Dict

import networkx as nx

from ..raw.sentence import Sentence
from ..target.target import Target
from ..target.mixin import TargetMixin


class ParsedSentence(TargetMixin):
    """
    Attributes
    ----------
    graph : nx.DiGraph
        dependency tree of sentence
    id2word : dict
        dictionary of all tokens
        value = token
        key = it's index
    id2lemma : dict
        dictionary of several tokens
        value = node of dependency tree in format: `lemma_pos`
        key = it's index
    id2dep : dict
        dictionary of dependency relations
        value = relation type
        key = node index
    """
    def __init__(self,
                 graph: nx.DiGraph,
                 id2word: Dict[int, str],
                 id2lemma: Dict[int, str],
                 id2dep: Dict[int, str],
                 id2init_id: Dict[int, int],
                 targets: List[Target] = None):
        if targets is None:
            targets = []
        super().__init__(targets=targets)

        self.graph = graph
        self.id2word = id2word
        self.id2lemma = id2lemma
        self.id2dep = id2dep

        # backward compatibility
        self.id2init_id = id2init_id

    def __len__(self):
        """Number of parsed tokens

        According to number of nodes in dependency tree"""
        return len(self.graph)

    def is_known(self, word_index: int) -> bool:
        """Is word with that index in known words"""
        return word_index in self.id2lemma

    def nodes_sentence_order(self) -> List[int]:
        """Nodes indexes in sentence order
        
        Returns
        -------
        nodes : list
            list of node's id
        """

        return [
            node_id for node_id, _ in sorted(self.id2init_id.items(), key=lambda item: item[1])
            if node_id in self.id2lemma
        ]

    def reset_targets(self):
        """Reset list of targets"""
        self.targets = []

    def reset_targets_polarities(self):
        """Reset only polarities of targets"""
        for target in self.targets:
            target.reset_polarity()

    def is_targets_contain_unknown(self) -> bool:
        for target in self.targets:
            for node in target.nodes:
                if not self.is_known(node):
                    return False
        return True

    def to_sentence(self) -> Sentence:
        """Convert to instance of Sentence class

        Returns
        -------
        sentence : Sentence
        """

        text = []
        for parsed_node_id, _ in sorted(self.id2init_id.items(), key=lambda item: item[1]):
            if parsed_node_id in self.id2word:
                text.append(self.id2word[parsed_node_id])

        targets = []
        for target in self.targets:
            target_nodes = []
            for parsed_node_id in target.nodes:
                target_nodes.append(self.id2init_id[parsed_node_id])
            targets.append(
                Target(nodes=target_nodes, category=target.category, polarity=target.polarity))
        return Sentence(text=text, targets=targets)

    def to_specified_sentence(self, text: List[str]) -> Sentence:
        """Convert to instance of Sentence class with specified text

        Returns
        -------
        sentence : Sentence
        """

        targets = []
        if self.targets:
            for target in self.targets:
                nodes = []
                for node in target.nodes:
                    node = self.id2init_id[node]
                    if node not in nodes:
                        nodes.append(node)
                targets.append(
                    Target(nodes=nodes, category=target.category, polarity=target.polarity))

        return Sentence(text=text, targets=targets)
