from absa import competition_path
from absa.utils.logging import configure_logging
from absa.requisites import upload_requisites
from absa.utils.embedding import Embeddings
from absa.preprocess.pipeline import preprocess_file
from absa.models.level.sentence.aspect.classifier import AspectClassifier as SentenceAspectClassifier
from absa.models.level.opinion.aspect.classifier import AspectClassifier as OpinionAspectClassifier
from absa.models.level.opinion.polarity.classifier import PolarityClassifier as OpinionPolarityClassifier

if __name__ == "__main__":
    configure_logging()
    upload_requisites()

    train_reviews = preprocess_file(pathway=competition_path.joinpath('train.xml'),
                                    using_dump=True)
    test_reviews = preprocess_file(pathway=competition_path.joinpath('test.xml'),
                                   using_dump=True)

    vocabulary = Embeddings.vocabulary
    emb_matrix = Embeddings.embeddings_matrix

    test_reviews_pred = SentenceAspectClassifier().fit_predict_score(
        vocabulary=vocabulary,
        emb_matrix=emb_matrix,
        train_reviews=train_reviews,
        test_reviews=test_reviews,
        save_state=True)

    OpinionAspectClassifier(vocabulary=vocabulary, emb_matrix=emb_matrix).fit_predict_score(
        train_reviews=train_reviews,
        test_reviews=test_reviews,
        test_reviews_pred=test_reviews_pred,
        save_state=True)
    OpinionPolarityClassifier(vocabulary=vocabulary, emb_matrix=emb_matrix).fit_predict_score(
        train_reviews=train_reviews, test_reviews=test_reviews, save_state=True)
