import os
import pickle
from typing import Any
import logging


def make_dump(obj: Any, pathway: str) -> None:
    directory = os.path.dirname(pathway)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(pathway, 'wb') as f:
        pickle.dump(obj, f)
    logging.info(f'Dump created: {pathway}')


def load_dump(pathway: str) -> Any:
    with open(pathway, 'rb') as f:
        obj = pickle.load(f)
    logging.info(f'Upload from dump: {pathway}')
    return obj
