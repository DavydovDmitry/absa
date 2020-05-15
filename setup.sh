# Install requirements
poetry shell
poetry install --no-dev
printf 'y\n\n' | python -c 'import stanfordnlp; stanfordnlp.download("ru")'

# Download embeddings
wget http://vectors.nlpl.eu/repository/20/185.zip -P ./embeddings/
unzip embeddings/185.zip -d embeddings/tayga_upos_skipgram_300_2_2019
rm embeddings/185.zip
rm embeddings/README

# Download dataset
mkdir ./datasets/SemEval2016/
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1RZUyBrWQ0OwlIsmN0axewKg21koYmgQf' -O ./datasets/SemEval2016/train.xml
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1JR3gblfNXQHApmDzY4FCCjv_0wVug7dO' -O ./datasets/SemEval2016/test.xml
