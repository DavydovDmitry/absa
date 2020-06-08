from typing import List, Dict, Tuple

import networkx as nx

from .opinion import Opinion


class ParsedSentence:
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
                 text: str,
                 id2word: Dict[int, Tuple[int, int]],
                 id2lemma: Dict[int, str],
                 id2dep: Dict[int, str],
                 opinions: List[Opinion] = None):
        if opinions is None:
            opinions = []
        self.opinions = opinions

        self.graph = graph
        self.text = text
        self.id2word = id2word
        self.id2lemma = id2lemma
        self.id2dep = id2dep

    def __len__(self):
        """Number of parsed tokens

        According to number of nodes in dependency tree"""
        return len(self.graph)

    def reset_opinions(self):
        self.opinions = []

    def is_known(self, word_index: int) -> bool:
        """Is word with that index in known words"""
        return word_index in self.id2lemma

    def get_text(self) -> str:
        return self.text

    def get_nodes(self, start_index: int, stop_index: int) -> List[int]:
        nodes_index = []
        for word_index, (word_start, word_stop) in self.id2word.items():
            if (start_index <= word_start) and (word_stop <= stop_index):
                nodes_index.append(word_index)
        return nodes_index

    def nodes_sentence_order(self) -> List[int]:
        """Nodes indexes in sentence order
        
        Returns
        -------
        nodes : list
            list of node's id
        """

        return [
            node_id for node_id, _ in sorted(self.id2lemma.items(), key=lambda item: item[0])
        ]

    def reset_opinions_polarities(self):
        """Reset only polarities of targets"""
        for opinion in self.opinions:
            opinion.reset_polarity()

    def is_opinions_contain_unknown(self) -> bool:
        for opinion in self.opinions:
            for node in opinion.nodes:
                if not self.is_known(node):
                    return False
        return True

    # todo
    # def to_sentence(self) -> Sentence:
    #     """Convert to instance of Sentence class
    #
    #     Returns
    #     -------
    #     sentence : Sentence
    #     """
    #
    #     text = []
    #     for parsed_node_id, _ in sorted(self.id2init_id.items(), key=lambda item: item[1]):
    #         if parsed_node_id in self.id2word:
    #             text.append(self.id2word[parsed_node_id])
    #
    #     opinions = []
    #     for opinion in self.opinions:
    #         opinions.append(
    #             Opinion(nodes=[
    #                 self.id2init_id[parsed_node_id] for parsed_node_id in opinion.nodes
    #             ],
    #                     category=opinion.category,
    #                     polarity=opinion.polarity))
    #     return Sentence(text=text, opinions=opinions)

    # todo
    # def to_specified_sentence(self, text: List[str]) -> Sentence:
    #     """Convert to instance of Sentence class with specified text
    #
    #     Returns
    #     -------
    #     sentence : Sentence
    #     """
    #
    #     opinions = []
    #     for opinion in self.opinions:
    #         nodes = []
    #         for node in opinion.nodes:
    #             node = self.id2init_id[node]
    #             if node not in nodes:
    #                 nodes.append(node)
    #         opinions.append(
    #             Opinion(nodes=nodes, category=opinion.category, polarity=opinion.polarity))
    #
    #     return Sentence(text=text, opinions=opinions)
