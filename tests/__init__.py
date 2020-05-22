import os

# -------------------------- Constants -----------------------------------

# preprocess stages
RAW_POSTFIX = '.raw'
SPELL_POSTFIX = '.spell'
DEP_POSTFIX = '.dep'

# --------------------------- Pathways -----------------------------------
test_path = os.path.dirname(os.path.abspath(__file__))
samples_path = os.path.join(test_path, 'samples')
test_dumps_path = os.path.join(test_path, 'dumps')

SemEval2016_filename = 'SemEval2016.xml'
SemEval2016_pathway = os.path.join(samples_path, SemEval2016_filename)
