from absa.io.input.text import from_txt
from absa.preprocess.pipeline import preprocess_pipeline
from absa.classification.level.sentence.aspect.classifier import AspectClassifier as SentenceAspectClassifier
from absa.classification.level.opinion import AspectClassifier as OpinionAspectClassifier
from absa.classification.level.opinion import PolarityClassifier
from absa.io.output.semeval2016 import to_xml

if __name__ == "__main__":
    reviews = from_txt()
    reviews = preprocess_pipeline(texts=reviews)
    reviews = SentenceAspectClassifier.load_model().predict(texts=reviews)
    reviews = OpinionAspectClassifier.load_model().predict(texts=reviews)
    reviews = PolarityClassifier.load_model().predict(texts=reviews)
    to_xml(texts=reviews)
