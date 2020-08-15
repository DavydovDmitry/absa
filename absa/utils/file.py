import urllib
import pathlib
import sys
import shutil
import logging
import time

import requests
from tqdm import tqdm

from absa import PROGRESSBAR_COLUMNS_NUM


def download_file(url, file: pathlib.Path, chunk_size: int = 8192):
    file.parent.mkdir(parents=True, exist_ok=True)
    meta = urllib.request.urlopen(url).info()
    file_size = int(meta['Content-Length'])

    start_time = time.time()
    logging.info(f'Start upload file: \'{file.name}\'')
    with requests.get(url, stream=True) as r:
        r.raise_for_status()

        # with open(file, 'wb') as f:
        #     with tqdm(total=file_size // chunk_size,
        #               ncols=PROGRESSBAR_COLUMNS_NUM,
        #               file=sys.stdout) as progress_bar:
        #         for chunk in r.iter_content(chunk_size=chunk_size):
        #             f.write(chunk)
        #             progress_bar.update(1)
        with open(file, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
    logging.info(f'Upload is complete. Elapsed time {(time.time() - start_time):.3f}s')
