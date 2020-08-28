import urllib
import zipfile
import pathlib
import logging

from absa import pretrained_embeddings_url, embeddings_dir, competition_path
from absa.utils.download import download_file


def _download_embeddings():
    """Upload archive with embeddings"""

    # check file presence
    embeddings_bin = embeddings_dir.joinpath('model.bin')
    if embeddings_bin.exists():
        logging.warning(f'File: \'{embeddings_bin}\' already exist. Skip download embeddings.')
        return

    # if file not found upload it
    embed_archive_filename = urllib.parse.urlparse(pretrained_embeddings_url).path.split(
        '/')[-1]
    download_file(url=pretrained_embeddings_url,
                  file=embeddings_dir.joinpath(embed_archive_filename))

    # extract archive
    archive_filename = embeddings_dir.joinpath(embed_archive_filename)
    with zipfile.ZipFile(archive_filename, 'r') as zip_ref:
        zip_ref.extractall(embeddings_dir)
    pathlib.Path(archive_filename).unlink()


def _download_train_dataset():
    train_url = 'https://drive.google.com/uc?export=download&id=1RZUyBrWQ0OwlIsmN0axewKg21koYmgQf'
    test_url = 'https://drive.google.com/uc?export=download&id=1JR3gblfNXQHApmDzY4FCCjv_0wVug7dO'

    for url, filename in zip([train_url, test_url], ['train.xml', 'test.xml']):
        file = competition_path.joinpath(filename)
        if file.exists():
            logging.warning(f'File: \'{file}\' already exist. Skip download dataset.')
        else:
            download_file(url=url, file=file)


def _download_ru_stanfordnlp_model():
    import stanfordnlp

    if not pathlib.Path.home().joinpath(
            'stanfordnlp_resources/ru_syntagrus_models/ru_syntagrus_tokenizer.pt').exists():
        stanfordnlp.download('ru')
    else:
        logging.warning(f'ru language model already exist.')


def upload_requisites():
    _download_embeddings()
    _download_train_dataset()
    _download_ru_stanfordnlp_model()
