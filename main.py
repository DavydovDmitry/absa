import urllib
import logging

from absa import pretrained_embeddings_url, embeddings_dir
from absa.utils.file import download_file
from absa.utils.logging import configure_logging

if __name__ == '__main__':
    configure_logging()
    logging.info('Start')
    filename = urllib.parse.urlparse(pretrained_embeddings_url).path.split('/')[-1]
    download_file(url=pretrained_embeddings_url, file=embeddings_dir.joinpath(filename))
