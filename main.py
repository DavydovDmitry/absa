import urllib
import zipfile
import pathlib

from absa import pretrained_embeddings_url, embeddings_dir
from absa.utils.download import download_file
from absa.utils.logging import configure_logging

if __name__ == '__main__':
    configure_logging()

    # upload archive with embeddings
    embed_archive_filename = urllib.parse.urlparse(pretrained_embeddings_url).path.split(
        '/')[-1]
    download_file(url=pretrained_embeddings_url,
                  file=embeddings_dir.joinpath(embed_archive_filename))

    # extract archive
    archive_filename = embeddings_dir.joinpath(embed_archive_filename)
    if zipfile.is_zipfile(archive_filename):
        with zipfile.ZipFile(archive_filename, 'r') as zip_ref:
            zip_ref.extractall(embeddings_dir)
        pathlib.Path(archive_filename).unlink()
