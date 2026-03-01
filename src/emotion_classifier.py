"""
Emotion Classifier using fine-tuned DistilBERT.

This module provides the EmotionClassifier class which loads a pre-trained
DistilBERT model fine-tuned on the GoEmotions dataset for 7-class emotion
classification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

logger = logging.getLogger(__name__)

EMOTION_LABELS = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "neutral",
    "sadness",
    "surprise",
]

DEFAULT_MODEL_NAME = "bhadresh-savani/distilbert-base-uncased-emotion"


@dataclass
class EmotionResult:
    """Container for emotion classification results."""

    text: str
    emotion: str
    confidence: float
    all_scores: dict[str, float]

    def __repr__(self) -> str:
        return (
            f"EmotionResult(emotion='{self.emotion}', "
            f"confidence={self.confidence:.4f})"
        )


class EmotionClassifier:
    """
    Fine-tuned DistilBERT emotion classifier.

    Classifies input text into one of 7 emotion categories:
    anger, disgust, fear, joy, neutral, sadness, surprise.

    Args:
        model_name: HuggingFace model identifier or local path.
        device: Compute device ('cpu', 'cuda', 'mps'). Auto-detected if None.
        max_length: Maximum tokenization length (default: 128).
        confidence_threshold: Minimum confidence to return a prediction.

    Example:
        >>> classifier = EmotionClassifier()
        >>> result = classifier.predict("I feel absolutely wonderful today!")
        >>> print(result.emotion, result.confidence)
        joy 0.9412
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
        max_length: int = 128,
        confidence_threshold: float = 0.0,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.confidence_threshold = confidence_threshold
        self.device = self._resolve_device(device)

        logger.info(f"Loading model '{model_name}' on device '{self.device}'")
        self._load_model()

    def _resolve_device(self, device: Optional[str]) -> str:
        if device is not None:
            return device
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_model(self) -> None:
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.model_name
        )
        self.model.to(self.device)
        self.model.eval()
        logger.info("Model loaded successfully.")

    @torch.no_grad()
    def predict(self, text: str) -> EmotionResult:
        """
        Predict emotion from a single text string.

        Args:
            text: Input text to classify.

        Returns:
            EmotionResult with predicted emotion and confidence scores.
        """
        if not text or not text.strip():
            raise ValueError("Input text must be a non-empty string.")

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        logits = self.model(**encoding).logits
        probabilities = F.softmax(logits, dim=-1).squeeze().cpu().tolist()

        id2label = self.model.config.id2label
        all_scores = {
            id2label[i].lower(): round(prob, 6)
            for i, prob in enumerate(probabilities)
        }

        predicted_id = int(torch.argmax(logits, dim=-1).item())
        emotion = id2label[predicted_id].lower()
        confidence = probabilities[predicted_id]

        return EmotionResult(
            text=text,
            emotion=emotion,
            confidence=round(confidence, 6),
            all_scores=all_scores,
        )

    def predict_batch(self, texts: list[str]) -> list[EmotionResult]:
        """
        Predict emotions for a batch of texts.

        Args:
            texts: List of input texts to classify.

        Returns:
            List of EmotionResult objects.
        """
        return [self.predict(text) for text in texts]

    @property
    def labels(self) -> list[str]:
        """Return list of emotion label names."""
        return list(self.model.config.id2label.values())

    def __repr__(self) -> str:
        return (
            f"EmotionClassifier(model='{self.model_name}', "
            f"device='{self.device}', labels={self.labels})"
        )
