import urllib
import zipfile
import pathlib

from absa import pretrained_embeddings_url, embeddings_dir, competition_path
from absa.utils.download import download_file
from absa.utils.logging import configure_logging


def upload_embeddings():
    """Upload archive with embeddings"""
    embed_archive_filename = urllib.parse.urlparse(pretrained_embeddings_url).path.split(
        '/')[-1]
    download_file(url=pretrained_embeddings_url,
                  file=embeddings_dir.joinpath(embed_archive_filename))

    # extract archive
    archive_filename = embeddings_dir.joinpath(embed_archive_filename)
    with zipfile.ZipFile(archive_filename, 'r') as zip_ref:
        zip_ref.extractall(embeddings_dir)
    pathlib.Path(archive_filename).unlink()


def upload_train_dataset():
    train_url = 'https://drive.google.com/uc?export=download&id=1RZUyBrWQ0OwlIsmN0axewKg21koYmgQf'
    test_url = 'https://drive.google.com/uc?export=download&id=1JR3gblfNXQHApmDzY4FCCjv_0wVug7dO'

    for url, file in zip([train_url, test_url], ['train.xml', 'test.xml']):
        download_file(url=url, file=competition_path.joinpath(file))


if __name__ == '__main__':
    configure_logging()
    upload_embeddings()
    upload_train_dataset()
