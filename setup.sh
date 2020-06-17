# Install requirements
poetry shell
poetry install --no-dev
printf 'y\n\n' | python -c 'import stanfordnlp; stanfordnlp.download("ru")'

# Download embeddings
mkdir -p ~/.absa/embeddings/
cd ~/.absa/embeddings/ &&
axel http://vectors.nlpl.eu/repository/20/185.zip -a -n 8 &&
unzip 185.zip -d tayga_upos_skipgram_300_2_2019 &&
rm 185.zip

# Download dataset
mkdir -p ~/.absa/datasets/SemEval2016/ &&
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1RZUyBrWQ0OwlIsmN0axewKg21koYmgQf' -O ./datasets/SemEval2016/train.xml &&
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1JR3gblfNXQHApmDzY4FCCjv_0wVug7dO' -O ./datasets/SemEval2016/test.xml

