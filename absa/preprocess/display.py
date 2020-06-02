"""
Module not longer used but can be useful for dep tree illustration.
"""

import logging
from typing import List
import pickle
import datetime
import math

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from typing_extensions import TypedDict

from absa import labeled_reviews_dump_path
from absa import EXTENDED_POLARITIES
from absa.text.raw.review import Review
from absa.text.parsed.sentence import ParsedSentence

NODE_SIZE = 500
ARROW_COLOR = '#e0e0e0'
TEXT_COLORS = {
    'NEUTRAL': '#FFB10C',
    'POSITIVE': '#26FF00',
    'NEGATIVE': '#FF002C',
}
NODE_COLORS = {
    'NEUTRAL': '#FFE985',
    'POSITIVE': '#DDFEC7',
    'VERY_POSITIVE': '#AFFF8A',
    'TOTALLY_POSITIVE': '#87FF27',
    'TOTALLY_NEGATIVE': '#FF0000',
    'VERY_NEGATIVE': '#FF6F54',
    'NEGATIVE': '#FFB9A9',
}


class TreeViewer:
    """Display dependency tree and paint nodes according to it's sentiment
     polarity.

     Also handle customer events to select sentence, paint nodes and save
     changes (make a dump of sentences).

     Parameters
     ----------
     parsed_reviews: List[List[ParsedSentence]]
        Essential parameter, exactly `parsed_reviews` keep polarity marks.
     reviews: List[Review]
        This parameter is using only for printing sentence text in the
        head of plot.
     """
    def __init__(self,
                 parsed_reviews: List[List[ParsedSentence]],
                 reviews: List[Review],
                 ax,
                 title_ax,
                 fig,
                 review_index=0,
                 sentence_index=0):
        self.review_index = review_index
        self.sentence_index = sentence_index
        self.ax = ax
        self.title_ax = title_ax
        self.fig = fig
        self.reviews = reviews
        self.parsed_reviews = parsed_reviews
        self.pos = None

        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_keypress)

    def on_keypress(self, event: matplotlib.backend_bases.MouseEvent):
        """Keyboard handle press.

        Using a, d, w, s for navigation through reviews sentences.
        """
        if str(event.key) == 'a':
            self.prev_image()
        elif str(event.key) == 'd':
            self.next_image()
        else:
            if str(event.key) == 'w':
                self.review_index = (self.review_index + 1) % len(self.reviews)
                self.sentence_index = 0
            elif str(event.key) == 's':
                self.review_index = int(math.fabs(self.review_index - 1)) % len(self.reviews)
                self.sentence_index = 0
            self.display_graph()

    def on_click(self, event: matplotlib.backend_bases.MouseEvent):
        """Click event handler"""

        node_radius = 900

        if event.xdata and event.ydata:
            if isinstance(event.inaxes, matplotlib.axes._subplots.Subplot):
                button_name = str(event.button).split('.')[-1]

                for index, (node_x, node_y) in self.pos.items():
                    node_x, node_y = self.ax.transData.transform((node_x, node_y))
                    x, y = self.ax.transData.transform((event.xdata, event.ydata))
                    if (node_x - x)**2 + (node_y - y)**2 < node_radius:
                        parsed_sentence = self.parsed_reviews[self.review_index][
                            self.sentence_index]
                        if button_name == 'LEFT':
                            self.parsed_reviews[self.review_index][
                                self.sentence_index].id2polarity_id[index] = (
                                    parsed_sentence.id2polarity_id[index] +
                                    1) % len(EXTENDED_POLARITIES)
                        else:
                            self.parsed_reviews[self.review_index][
                                self.sentence_index].id2polarity_id[index] = (
                                    parsed_sentence.id2polarity_id[index] -
                                    1) % len(EXTENDED_POLARITIES)
                        polarity = parsed_sentence.id2polarity_id[index]
                        color_name = [
                            k for k, v in EXTENDED_POLARITIES.items() if v == polarity
                        ][0]
                        nx.draw_networkx_nodes(parsed_sentence.graph,
                                               self.pos,
                                               nodelist=[
                                                   index,
                                               ],
                                               node_size=NODE_SIZE,
                                               node_color=NODE_COLORS[color_name],
                                               ax=self.ax)

    def display_graph(self):
        """Display dependency tree of sentence"""
        def split_text(s: str, line_length=100):
            list_text = ['']
            for chunk in s.split(' '):
                if len(token) + len(list_text[-1]) > line_length:
                    list_text.append(chunk + ' ')
                else:
                    list_text[-1] = list_text[-1] + chunk + ' '
            return '\n'.join(list_text)

        parsed_sentence = self.parsed_reviews[self.review_index][self.sentence_index]
        G = parsed_sentence.graph
        labels = {
            k: v
            for k, v in parsed_sentence.id2word.items() if k in parsed_sentence.id2lemma
        }

        self.ax.cla()
        self.pos = graphviz_layout(G, prog='dot')
        nx.draw(G,
                pos=self.pos,
                labels=labels,
                with_labels=True,
                node_size=NODE_SIZE,
                node_color=NODE_COLORS['NEUTRAL'],
                font_size=7,
                edge_color=ARROW_COLOR,
                ax=self.ax)

        # edge labels
        nx.draw_networkx_edge_labels(
            G,
            self.pos,
            font_size=7,
            edge_labels={edge: parsed_sentence.id2dep[edge[0]]
                         for edge in G.edges},
            ax=self.ax,
        )

        # group nodes by polarities
        node_polarities = {polarity_id: list() for polarity_id in EXTENDED_POLARITIES.values()}
        for node_id, polarity_id in parsed_sentence.id2polarity_id.items():
            node_polarities[polarity_id].append(node_id)
        for polarity in EXTENDED_POLARITIES:
            if node_polarities[EXTENDED_POLARITIES[polarity]]:
                nx.draw_networkx_nodes(G,
                                       self.pos,
                                       nodelist=node_polarities[EXTENDED_POLARITIES[polarity]],
                                       node_size=NODE_SIZE,
                                       node_color=NODE_COLORS[polarity],
                                       ax=self.ax)
        sentence_text = self.reviews[self.review_index].sentences[self.sentence_index].text

        # Plot color sentence text
        self.title_ax.clear()
        self.title_ax.axis('off')

        y_step = 0.35
        x = 0
        y = 0.5
        self.title_ax.text(
            x,
            y,
            f'{self.review_index}/{len(self.reviews)} review.   ' +
            f'{self.sentence_index}/{len(self.reviews[self.review_index].sentences)} sentence.',
            color='black')
        y -= y_step

        max_line_length = 120
        line_length = 0
        for token_index, token in enumerate(sentence_text):
            color = 'black'
            for target in self.reviews[self.review_index].sentences[
                    self.sentence_index].opinions:
                if token_index in target.nodes:
                    color = TEXT_COLORS[target.polarity.upper()]

            text = self.title_ax.text(x, y, token + " ", color=color)
            ex = text.get_window_extent()
            x, _ = self.title_ax.transData.inverted().transform(
                [ex.intervalx[1], ex.intervaly[0]])
            line_length += len(token)
            if line_length >= max_line_length:
                y -= y_step
                x = 0
                line_length = 0
        self.ax.figure.canvas.draw()

    def button_press(self, event: matplotlib.backend_bases.MouseEvent):
        """Handler of mouse button press"""

        text = event.inaxes.texts[0]._text
        if text == 'Previous':
            self.prev_image()
        if text == 'Next':
            self.next_image()

    def next_image(self):
        """Display next one sentence."""

        self.sentence_index += 1
        if self.sentence_index >= len(self.parsed_reviews[self.review_index]):
            self.sentence_index = 0
            self.review_index += 1
            if self.review_index >= len(self.parsed_reviews):
                self.review_index = 0
        self.display_graph()

    def prev_image(self):
        """Display previous sentence."""

        self.sentence_index -= 1
        if self.sentence_index < 0:
            if self.review_index > 0:
                self.review_index -= 1
                self.sentence_index = len(self.parsed_reviews[self.review_index]) - 1
            else:
                self.sentence_index = 0
        self.display_graph()

    def input_review_index(self, input_str: str):
        try:
            review_index, sentence_index = map(lambda x: int(x), input_str.split('.'))
        except ValueError:
            pass
        else:
            if len(self.reviews) > review_index:
                self.review_index = review_index
            if len(self.reviews[review_index].sentences) > sentence_index:
                self.sentence_index = sentence_index
            self.display_graph()

    def dump_labeled_reviews(self, *args):
        dump = {
            'review_index': self.review_index,
            'sentence_index': self.sentence_index,
            'parsed_reviews': self.parsed_reviews
        }
        with open(labeled_reviews_dump_path, 'wb') as f:
            pickle.dump(dump, f)
        logging.info(f'{datetime.datetime.now().ctime()}. Dump of labeled reviews is made.')


class LabeledReviewsDump(TypedDict):
    review_index: int
    sentence_index: int
    parsed_reviews: List[List[ParsedSentence]]


def load_labeled_reviews(file_pathway) -> LabeledReviewsDump:
    with open(file_pathway, 'rb') as f:
        dump = pickle.load(f)
    logging.info(f'{datetime.datetime.now().ctime()}. Upload labeled reviews.')
    return dump


def view_dep_tree(reviews: List[Review],
                  parsed_reviews: List[List[ParsedSentence]],
                  review_index=0,
                  sentence_index=0):
    """Pass reviews to viewer..."""
    matplotlib.use('Qt5Agg')
    plt.rcParams['keymap.save'].remove('s')
    fig, ax = plt.subplots()

    # prev/next buttons
    plt.subplots_adjust(bottom=0.2)

    title_ax = plt.axes([0.05, 0.95, 0.9, 0.05])

    callback = TreeViewer(reviews=reviews,
                          parsed_reviews=parsed_reviews,
                          ax=ax,
                          title_ax=title_ax,
                          fig=fig,
                          review_index=review_index,
                          sentence_index=sentence_index)

    prev_left = 0.8
    next_left = 0.9
    button_bottom = 0.02
    button_width = 0.06
    button_height = 0.04

    # prev / next
    ax_prev = plt.axes([prev_left, button_bottom, button_width, button_height])
    ax_next = plt.axes([next_left, button_bottom, button_width, button_height])
    b_next = Button(ax_next, 'Next')
    b_prev = Button(ax_prev, 'Previous')
    b_next._click = callback.next_image
    b_next.connect_event('button_press_event', callback.button_press)

    # save
    ax_save = plt.axes([0.5, button_bottom, button_width, button_height])
    b_save = Button(ax_save, 'Save')
    b_save.on_clicked(callback.dump_labeled_reviews)

    # input
    ax_review_index = plt.axes([0.1, button_bottom, button_width, button_height])
    text_box = TextBox(ax_review_index, 'Review.Sentence', initial='0.0')
    text_box.on_submit(callback.input_review_index)

    # Maximise the plotting window
    try:
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
    except AttributeError as e:
        logging.warning(e)
    plt.show()
