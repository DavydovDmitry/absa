"""Specify constants and pathways here"""
import os

# ------------------------------- Constants -----------------------------------
SCORE_DECIMAL_LEN = 5
PROGRESSBAR_COLUMNS_NUM = 100

# ------------------------------- Embeddings ----------------------------------
UNKNOWN_WORD = '<unk>'  # index of that word will be using if word not found in vocabulary
PAD_WORD = '<pad>'  # index of that word will be using to pad sentence to length to maximal length in batch

# ------------------------------- Pathways ------------------------------------
module_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
competition_path = os.path.join(module_path, 'datasets/SemEval2016')

TEST_APPENDIX = '.test'  # suffix for test dumps

train_reviews_path = os.path.join(competition_path, 'dataset', 'train.xml')
test_reviews_path = os.path.join(competition_path, 'dataset', 'test.xml')

embeddings_path = os.path.join(module_path, 'embeddings')
word2vec_model_path = os.path.join(embeddings_path, 'tayga_upos_skipgram_300_2_2019',
                                   'model.bin')

images_path = os.path.join(module_path, 'images')
log_path = os.path.join(module_path, 'logs')

# --------------------------------- Dumps -------------------------------------
dumps_path = os.path.join(module_path, 'dumps')

checked_reviews_dump_path = os.path.join(dumps_path, 'checked_reviews')
parsed_reviews_dump_path = os.path.join(dumps_path, 'dep_parsed_sentence')
labeled_reviews_dump_path = os.path.join(dumps_path, 'labeled_reviews')

sb12_classifier_path = os.path.join(dumps_path, 'sb12_classifier')
sb12_train_data_path = os.path.join(dumps_path, 'sb12_train_data')
sb12_test_data_path = os.path.join(dumps_path, 'sb12_test_data')

aspect_classifier_dump_path = os.path.join(dumps_path, 'aspect_classifier.pt')
polarity_classifier_dump_path = os.path.join(dumps_path, 'polarity_classifier.pt')
