"""Specify constants, pathways, dumps"""

import os

# ------------------------------- Constants -----------------------------------
SCORE_DECIMAL_LEN = 5
PROGRESSBAR_COLUMNS_NUM = 100

# ------------------------------- Embeddings ----------------------------------
UNKNOWN_WORD = '<unk>'  # index of word that not found in vocabulary
PAD_WORD = '<pad>'  # index of that word will be using to pad sentence to maximal length in batch

# ------------------------------- Pathways ------------------------------------
module_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
competition_path = os.path.join(module_path, 'datasets/SemEval2016')

TEST_APPENDIX = '.test'  # suffix for test dumps

train_reviews_path = os.path.join(competition_path, 'train.xml')
test_reviews_path = os.path.join(competition_path, 'test.xml')

embeddings_path = os.path.join(module_path, 'embeddings')
word2vec_model_path = os.path.join(embeddings_path, 'tayga_upos_skipgram_300_2_2019',
                                   'model.bin')

images_path = os.path.join(module_path, 'analysis', 'images')
log_path = os.path.join(module_path, 'logs')

# --------------------------------- Dumps -------------------------------------
dumps_path = os.path.join(module_path, 'dumps')
classifiers_dump_path = os.path.join(dumps_path, 'classifiers')
data_dump_path = os.path.join(dumps_path, 'data')

vocabulary_dump = os.path.join(dumps_path, 'vocabulary')
embed_matrix_path = os.path.join(dumps_path, 'embed_matrix')

# Processed data
raw_reviews_dump_path = os.path.join(data_dump_path, 'reviews')
checked_reviews_dump_path = os.path.join(data_dump_path, 'checked_reviews')
parsed_reviews_dump_path = os.path.join(data_dump_path, 'dep_parsed_sentence')

# classifiers
sentence_aspect_classifier_dump_path = os.path.join(classifiers_dump_path,
                                                    'sentence_aspect_classifier.pt')
opinion_aspect_classifier_dump_path = os.path.join(classifiers_dump_path,
                                                   'opinion_aspect_classifier.pt')
opinion_polarity_classifier_dump_path = os.path.join(classifiers_dump_path,
                                                     'opinion_polarity_classifier.pt')
