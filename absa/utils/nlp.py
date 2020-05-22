"""NLP pipeline"""

import sys
import os
import warnings
import time

import stanfordnlp


class MetaNLPPipeline:
    def __init__(self, *args, **kwargs):
        sys.stdout = open(os.devnull, 'w')
        while True:
            try:
                self._nlp = stanfordnlp.Pipeline(lang='ru', )
            except RuntimeError:
                time.sleep(1)
            else:
                break
        warnings.filterwarnings("ignore", category=UserWarning)
        sys.stdout = sys.__stdout__

    @property
    def nlp(self):
        return self._nlp


class NLPPipeline(metaclass=MetaNLPPipeline):
    pass
