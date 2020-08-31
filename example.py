from absa.io.input.text import from_txt
from absa.preprocess.pipeline import preprocess_texts
from absa.models.level.sentence.aspect.classifier import AspectClassifier as SentenceAspectClassifier
from absa.models.level.opinion.aspect.classifier import AspectClassifier as OpinionAspectClassifier
from absa.models.level.opinion.polarity.classifier import PolarityClassifier
from absa.io.output.semeval2016 import to_xml

if __name__ == "__main__":
    reviews = from_txt()
    reviews = preprocess_texts(texts=reviews)
    reviews = SentenceAspectClassifier.load_model().predict(texts=reviews)
    reviews = OpinionAspectClassifier.load_model().predict(texts=reviews)
    reviews = PolarityClassifier.load_model().predict(texts=reviews)
    to_xml(texts=reviews)
