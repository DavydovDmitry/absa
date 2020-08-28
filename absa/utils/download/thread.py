import urllib
import pathlib
import shutil
import logging
import time
import multiprocessing
from math import ceil
import functools
import string

import requests


def download_chunk(chunk_index: int,
                   chunk_size: int,
                   url: str,
                   chunk_file_template=string.Template('chunk-$chunk_size.bin')):
    """Download specified chunk of file [chunk_index*chunk_size:(chunk_index+1)*(chunk_size-1)] and save in file.

    Parameters
    ----------
    chunk_index
        number of chunk to calculate what range to request
    chunk_size
        size of chunk in bytes to requests
    url
        url of file to upload
    chunk_file_template
        string template for filepath where save downloaded chunk
    """

    with open(chunk_file_template.substitute(chunk_index=chunk_index), 'wb') as f:
        with requests.get(
                url=url,
                headers={
                    'Range':
                    f'bytes={chunk_index * chunk_size}-{(chunk_index + 1) * chunk_size - 1}'
                },
                stream=True) as r:
            logging.info(f'Start upload {chunk_index} chunk... HTTP Status {r.status_code}')
            shutil.copyfileobj(r.raw, f)


def download_file(url: str, file: pathlib.Path, num_chunks: int = multiprocessing.cpu_count()):
    """Upload file in threads by chunks

    Parameters
    ----------
    url
        url of file to upload
    file
        pathway where save downloaded file
    num_chunks
    """
    chunk_file_template = string.Template(
        str(file.parent.joinpath(file.name + '-$chunk_index')))

    # retrieve info about file size
    file.parent.mkdir(parents=True, exist_ok=True)
    meta = urllib.request.urlopen(url).info()
    file_size = int(meta['Content-Length'])

    # upload file in parallel by chunks
    # todo: progress bar for upload
    start_time = time.time()
    logging.info(f'Start upload file: \'{file.name}\'')
    with multiprocessing.Pool(num_chunks) as pool:
        results = pool.map(
            functools.partial(download_chunk,
                              url=url,
                              chunk_size=ceil(file_size / num_chunks),
                              chunk_file_template=chunk_file_template), range(num_chunks))
    logging.info(f'Upload is complete. Elapsed time {(time.time() - start_time):.3f}s')

    # concatenate chunks to target filename
    with open(file, 'wb') as target_stream:
        for chunk_index in range(num_chunks):
            chunk_filename = chunk_file_template.substitute(chunk_index=chunk_index)
            with open(chunk_filename, 'rb') as chunk_stream:
                target_stream.write(chunk_stream.read())

    # rm chunks files
    for chunk_index in range(num_chunks):
        pathlib.Path(chunk_file_template.substitute(chunk_index=chunk_index)).unlink()
