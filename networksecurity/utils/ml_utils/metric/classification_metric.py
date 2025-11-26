from networksecurity.entity.artifact_entity import ClassificationMetricArtifact
from networksecurity.exception.exception import NetworkSecurityException
import sys
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def get_classification_score(y_true, y_pred) -> ClassificationMetricArtifact:
    try:
        return ClassificationMetricArtifact(
            f1_score=f1_score(y_true, y_pred, average='weighted'),
            precision_score=precision_score(y_true, y_pred, average='weighted'),
            recall_score=recall_score(y_true, y_pred, average='weighted'),
            accuracy_score=accuracy_score(y_true, y_pred)
        )
    except Exception as e:
        raise NetworkSecurityException(e, sys)
