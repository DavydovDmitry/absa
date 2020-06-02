from absa import SCORE_DECIMAL_LEN


class Score:
    def __init__(self, precision: float, recall: float, f1: float):
        self.precision = precision
        self.recall = recall
        self.f1 = f1

    def __str__(self):
        score_len = SCORE_DECIMAL_LEN
        return '(' + \
               f'precision={self.precision:.{score_len}f}, ' + \
               f'recall={self.recall:.{score_len}f}, ' + \
               f'f1={self.f1:.{score_len}f}' + ')'
