import urllib
import pathlib
import logging
import time
from math import ceil
import asyncio

import aiohttp
import aiofiles


async def download_chunk(chunk_index: int, chunk_size: int, url: str, file: pathlib.Path):
    """Download specified chunk of file

    Download [chunk_index*chunk_size : (chunk_index+1)*(chunk_size-1)] bytes
    and write to file when chunk uploaded.

    Parameters
    ----------
    chunk_index
        number of chunk to calculate what range to request
    chunk_size
        size of chunk in bytes to requests
    url
        url of file to upload
    file : pathlib.Path
    """

    start_byte = chunk_index * chunk_size  # first byte
    end_byte = (chunk_index + 1) * chunk_size - 1  # last byte (included)
    headers = {'Range': f'bytes={start_byte}-{end_byte}'}
    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.get(url) as resp:
            data = await resp.content.read()

            # when chunk was uploaded open file and write to specific bytes
            async with aiofiles.open(file, mode='rb+') as f:
                await f.seek(start_byte, 0)
                await f.write(data)


async def schedule_download(url: str, file: pathlib.Path, num_chunks: int):
    """Schedule chunks to download"""

    # retrieve info about file size
    file.parent.mkdir(parents=True, exist_ok=True)
    meta = urllib.request.urlopen(url).info()
    file_size = int(meta['Content-Length'])

    # check can allocate file size
    with open(file, 'wb') as f:
        f.seek(file_size - 1)
        f.write(b'\0')

    logging.info(f'Start download file: \'{file.name}\'...')
    start_time = time.time()
    await asyncio.gather(*(download_chunk(
        chunk_index=chunk, chunk_size=ceil(file_size / num_chunks), url=url, file=file)
                           for chunk in range(num_chunks)))
    logging.info(f'Download is completed. Elapsed time: {(time.time() - start_time):.3f}')


def download_file(url: str, file: pathlib.Path, num_chunks: int = 50):
    """Upload file by chunks

    Parameters
    ----------
    url
        url of file to upload
    file
        pathway to save downloaded file
    num_chunks
    """

    file.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(schedule_download(url, file, num_chunks))
