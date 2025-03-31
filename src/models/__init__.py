"""Model definition and training modules."""

from src.models.classifier import (
    ExerciseDataset, 
    MultimodalExerciseClassifier,
    ExerciseClassifier
)

from src.models.trainer import ModelTrainer

__all__ = [
    'ExerciseDataset',
    'MultimodalExerciseClassifier',
    'ExerciseClassifier',
    'ModelTrainer'
]