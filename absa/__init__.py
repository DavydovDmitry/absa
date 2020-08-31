import os
import pathlib

# ------------------------------- Constants -----------------------------------
SCORE_DECIMAL_LEN = 5
PROGRESSBAR_COLUMNS_NUM = 100

# ------------------------------- Embeddings ----------------------------------
UNKNOWN_WORD = '<unk>'  # index of word that not found in vocabulary
PAD_WORD = '<pad>'  # index of that word will be using to pad sentence to maximal length in batch

# ------------------------------- Pathways ------------------------------------
project_path = pathlib.Path(__file__).resolve().parent

example_path = os.path.join(project_path, 'example')
input_path = os.path.join(example_path, 'input')
output_path = os.path.join(example_path, 'output')

images_path = os.path.join(project_path, 'analysis', 'images')

# ----------------------------- Static ----------------------------------------
static_files_path = pathlib.Path.home().joinpath('.absa')

competition_path = static_files_path.joinpath('datasets/SemEval2016')

pretrained_embeddings_url = 'http://vectors.nlpl.eu/repository/20/185.zip'
embeddings_dir = static_files_path.joinpath('embeddings')
embeddings_path = embeddings_dir.joinpath('model.bin')

log_path = static_files_path.joinpath('logs')

# --------------------------------- Dumps -------------------------------------
dumps_path = static_files_path.joinpath('dumps')
classifiers_dump_path = dumps_path.joinpath('classifiers')
data_dump_path = dumps_path.joinpath('data')

vocabulary_dump = dumps_path.joinpath('vocabulary')
embed_matrix_path = dumps_path.joinpath('embed_matrix')

# Processed data
raw_reviews_dump_path = data_dump_path.joinpath('reviews')
checked_reviews_dump_path = data_dump_path.joinpath('checked_reviews')
parsed_reviews_dump_path = data_dump_path.joinpath('dep_parsed_sentence')

# classifiers
sentence_aspect_classifier_dump_path = classifiers_dump_path.joinpath(
    'sentence_aspect_classifier.pt')
opinion_aspect_classifier_dump_path = classifiers_dump_path.joinpath(
    'opinion_aspect_classifier.pt')
opinion_polarity_classifier_dump_path = classifiers_dump_path.joinpath(
    'opinion_polarity_classifier.pt')
