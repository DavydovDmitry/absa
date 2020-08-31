import os
import pickle
from typing import Any, Union
import logging
import pathlib

from absa import dumps_path


def dump_is_exist(file):
    return dumps_path.joinpath(file).is_file()


def make_dump(obj: Any, pathway: Union[pathlib.Path, str]) -> None:
    """Create dump of object in file ../.absa/dumps/`pathway`

    obj
        object to store
    pathway
        path where to store
    """

    pathway = dumps_path.joinpath(pathway)
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
    """Restore obj from file ../.absa/dumps/`pathway`"""

    pathway = dumps_path.joinpath(pathway)
    with open(pathway, 'rb') as f:
        obj = pickle.load(f)
    logging.info(f'Upload from dump: {pathway}')
    return obj
