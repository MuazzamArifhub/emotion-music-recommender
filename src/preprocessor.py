"""
Text Preprocessor for Emotion Music Recommender.

Handles tokenization, cleaning, and normalization of raw input text
before passing it to the EmotionClassifier.
"""

from __future__ import annotations

import logging
import re
import string
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Contraction mapping for normalization
CONTRACTIONS: dict[str, str] = {
      "i'm": "i am",
      "i've": "i have",
      "i'll": "i will",
      "i'd": "i would",
      "you're": "you are",
      "you've": "you have",
      "you'll": "you will",
      "you'd": "you would",
      "he's": "he is",
      "she's": "she is",
      "it's": "it is",
      "we're": "we are",
      "we've": "we have",
      "we'll": "we will",
      "we'd": "we would",
      "they're": "they are",
      "they've": "they have",
      "they'll": "they will",
      "they'd": "they would",
      "can't": "cannot",
      "couldn't": "could not",
      "don't": "do not",
      "doesn't": "does not",
      "didn't": "did not",
      "won't": "will not",
      "wouldn't": "would not",
      "isn't": "is not",
      "aren't": "are not",
      "wasn't": "was not",
      "weren't": "were not",
      "haven't": "have not",
      "hasn't": "has not",
      "hadn't": "had not",
      "shouldn't": "should not",
      "mustn't": "must not",
      "that's": "that is",
      "there's": "there is",
      "what's": "what is",
      "let's": "let us",
}


@dataclass
class PreprocessedText:
      """Container for preprocessed text and metadata."""

    original: str
    cleaned: str
    word_count: int
    char_count: int
    was_truncated: bool = False

    def __repr__(self) -> str:
              return (
                            f"PreprocessedText(words={self.word_count}, "
                            f"truncated={self.was_truncated})"
              )


class TextPreprocessor:
      """
          Text preprocessing pipeline for emotion classification.

              Applies cleaning, normalization, and optional truncation to raw
                  text before it is fed to the EmotionClassifier.

                      Args:
                              lowercase: Convert text to lowercase (default: True).
                                      expand_contractions: Expand English contractions (default: True).
                                              remove_urls: Strip URLs from text (default: True).
                                                      remove_punctuation: Remove punctuation marks (default: False).
                                                              max_words: Truncate text to this many words (default: None).

                                                                  Example:
                                                                          >>> preprocessor = TextPreprocessor()
                                                                                  >>> result = preprocessor.process("I can't believe how amazing this is!")
                                                                                          >>> print(result.cleaned)
                                                                                                  i cannot believe how amazing this is!
                                                                                                      """

    def __init__(
              self,
              lowercase: bool = True,
              expand_contractions: bool = True,
              remove_urls: bool = True,
              remove_punctuation: bool = False,
              max_words: int | None = None,
    ) -> None:
              self.lowercase = lowercase
              self.expand_contractions = expand_contractions
              self.remove_urls = remove_urls
              self.remove_punctuation = remove_punctuation
              self.max_words = max_words
              logger.info("TextPreprocessor initialized.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, text: str) -> PreprocessedText:
              """
                      Run the full preprocessing pipeline on a single text string.

                              Args:
                                          text: Raw input text.

                                                  Returns:
                                                              PreprocessedText with cleaned text and metadata.

                                                                      Raises:
                                                                                  ValueError: If text is empty or None.
                                                                                          """
              if not text or not text.strip():
                            raise ValueError("Input text must be a non-empty string.")

              original = text
              cleaned = text

        if self.remove_urls:
                      cleaned = self._remove_urls(cleaned)

        if self.lowercase:
                      cleaned = cleaned.lower()

        if self.expand_contractions:
                      cleaned = self._expand_contractions(cleaned)

        cleaned = self._remove_extra_whitespace(cleaned)

        if self.remove_punctuation:
                      cleaned = self._remove_punctuation(cleaned)

        was_truncated = False
        if self.max_words is not None:
                      cleaned, was_truncated = self._truncate(cleaned, self.max_words)

        return PreprocessedText(
                      original=original,
                      cleaned=cleaned,
                      word_count=len(cleaned.split()),
                      char_count=len(cleaned),
                      was_truncated=was_truncated,
        )

    def process_batch(self, texts: list[str]) -> list[PreprocessedText]:
              """
                      Run preprocessing on a batch of texts.

                              Args:
                                          texts: List of raw input strings.

                                                  Returns:
                                                              List of PreprocessedText objects.
                                                                      """
              return [self.process(t) for t in texts]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _remove_urls(text: str) -> str:
              """Strip http/https URLs and www addresses from text."""
              return re.sub(r"https?://\S+|www\.\S+", "", text).strip()

    @staticmethod
    def _expand_contractions(text: str) -> str:
              """Expand common English contractions."""
              pattern = re.compile(
                  r"\b(" + "|".join(re.escape(k) for k in CONTRACTIONS) + r")\b",
                  re.IGNORECASE,
              )
              return pattern.sub(
                  lambda m: CONTRACTIONS[m.group(0).lower()], text
              )

    @staticmethod
    def _remove_extra_whitespace(text: str) -> str:
              """Collapse multiple spaces/tabs/newlines into a single space."""
              return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _remove_punctuation(text: str) -> str:
              """Remove all punctuation characters."""
              return text.translate(str.maketrans("", "", string.punctuation))

    @staticmethod
    def _truncate(text: str, max_words: int) -> tuple[str, bool]:
              """Truncate text to at most *max_words* words."""
              words = text.split()
              if len(words) <= max_words:
                            return text, False
                        return " ".join(words[:max_words]), True

    def __repr__(self) -> str:
              return (
                  f"TextPreprocessor(lowercase={self.lowercase}, "
                  f"expand_contractions={self.expand_contractions}, "
                  f"remove_urls={self.remove_urls})"
    )
