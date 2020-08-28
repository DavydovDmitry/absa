import os
import pickle
from typing import Any, Union
import logging
import pathlib


def make_dump(obj: Any, pathway: Union[pathlib.Path, str]) -> None:
    """Create dump of object

    obj
    pathway
    """
    if isinstance(pathway, pathlib.Path):
        pathway.parent.mkdir(parents=True, exist_ok=True)
    elif isinstance(pathway, str):
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
