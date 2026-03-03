"""
Emotion Music Recommender - Source Package.

An end-to-end ML pipeline that detects emotion from user text input
and recommends music tracks using a fine-tuned DistilBERT transformer.
"""

from .emotion_classifier import EmotionClassifier, EmotionResult
from .music_recommender import MusicRecommender, TrackRecommendation
from .preprocessor import TextPreprocessor

__all__ = [
      "EmotionClassifier",
      "EmotionResult",
      "MusicRecommender",
      "TrackRecommendation",
      "TextPreprocessor",
]

__version__ = "1.0.0"
__author__ = "MuazzamArifhub"
