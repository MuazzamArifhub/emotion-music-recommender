"""
Music Recommender Engine.

Maps detected emotions to curated music genre/mood/BPM profiles
and returns ranked track recommendations.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TrackRecommendation:
    """A single music track recommendation."""

    genre: str
    bpm: int
    mood: str
    energy: str
    valence: float  # 0.0 (negative) to 1.0 (positive)
    danceability: float  # 0.0 to 1.0
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "genre": self.genre,
            "bpm": self.bpm,
            "mood": self.mood,
            "energy": self.energy,
            "valence": self.valence,
            "danceability": self.danceability,
            "tags": self.tags,
        }


# Emotion -> Music profile mapping
EMOTION_PROFILES: dict[str, list[dict]] = {
    "joy": [
        {
            "genre": "Pop",
            "bpm_range": (120, 140),
            "mood": "Uplifting",
            "energy": "High",
            "valence": 0.85,
            "danceability": 0.80,
            "tags": ["happy", "upbeat", "cheerful"],
        },
        {
            "genre": "Dance / EDM",
            "bpm_range": (128, 145),
            "mood": "Euphoric",
            "energy": "Very High",
            "valence": 0.90,
            "danceability": 0.92,
            "tags": ["euphoric", "festival", "energetic"],
        },
        {
            "genre": "Funk",
            "bpm_range": (100, 120),
            "mood": "Groovy",
            "energy": "High",
            "valence": 0.88,
            "danceability": 0.85,
            "tags": ["funky", "groove", "feel-good"],
        },
    ],
    "sadness": [
        {
            "genre": "Indie Folk",
            "bpm_range": (60, 80),
            "mood": "Melancholic",
            "energy": "Low",
            "valence": 0.20,
            "danceability": 0.25,
            "tags": ["sad", "introspective", "acoustic"],
        },
        {
            "genre": "Blues",
            "bpm_range": (65, 85),
            "mood": "Heartfelt",
            "energy": "Low-Medium",
            "valence": 0.25,
            "danceability": 0.30,
            "tags": ["blues", "soulful", "emotional"],
        },
        {
            "genre": "Classical Piano",
            "bpm_range": (50, 70),
            "mood": "Tender",
            "energy": "Very Low",
            "valence": 0.30,
            "danceability": 0.15,
            "tags": ["piano", "classical", "delicate"],
        },
    ],
    "anger": [
        {
            "genre": "Metal",
            "bpm_range": (150, 200),
            "mood": "Intense",
            "energy": "Very High",
            "valence": 0.15,
            "danceability": 0.40,
            "tags": ["heavy", "aggressive", "powerful"],
        },
        {
            "genre": "Hard Rock",
            "bpm_range": (130, 160),
            "mood": "Fierce",
            "energy": "High",
            "valence": 0.20,
            "danceability": 0.50,
            "tags": ["rock", "intense", "driving"],
        },
        {
            "genre": "Drum and Bass",
            "bpm_range": (160, 180),
            "mood": "Aggressive",
            "energy": "Very High",
            "valence": 0.18,
            "danceability": 0.65,
            "tags": ["dnb", "fast", "adrenaline"],
        },
    ],
    "fear": [
        {
            "genre": "Ambient",
            "bpm_range": (50, 70),
            "mood": "Calm",
            "energy": "Very Low",
            "valence": 0.40,
            "danceability": 0.10,
            "tags": ["ambient", "soothing", "meditative"],
        },
        {
            "genre": "Classical",
            "bpm_range": (55, 75),
            "mood": "Soothing",
            "energy": "Low",
            "valence": 0.45,
            "danceability": 0.12,
            "tags": ["orchestral", "calming", "peaceful"],
        },
    ],
    "surprise": [
        {
            "genre": "Electronic / Glitch",
            "bpm_range": (110, 140),
            "mood": "Unexpected",
            "energy": "High",
            "valence": 0.65,
            "danceability": 0.70,
            "tags": ["electronic", "quirky", "experimental"],
        },
        {
            "genre": "Jazz Fusion",
            "bpm_range": (100, 130),
            "mood": "Playful",
            "energy": "Medium",
            "valence": 0.70,
            "danceability": 0.60,
            "tags": ["jazz", "fusion", "improvisational"],
        },
    ],
    "disgust": [
        {
            "genre": "Punk",
            "bpm_range": (140, 180),
            "mood": "Raw",
            "energy": "Very High",
            "valence": 0.12,
            "danceability": 0.55,
            "tags": ["punk", "rebellious", "raw"],
        },
        {
            "genre": "Grunge",
            "bpm_range": (120, 150),
            "mood": "Gritty",
            "energy": "High",
            "valence": 0.18,
            "danceability": 0.45,
            "tags": ["grunge", "distorted", "edgy"],
        },
    ],
    "neutral": [
        {
            "genre": "Lo-fi Hip Hop",
            "bpm_range": (70, 90),
            "mood": "Relaxed",
            "energy": "Low-Medium",
            "valence": 0.55,
            "danceability": 0.55,
            "tags": ["lofi", "chill", "study"],
        },
        {
            "genre": "Jazz",
            "bpm_range": (75, 95),
            "mood": "Easygoing",
            "energy": "Low-Medium",
            "valence": 0.58,
            "danceability": 0.48,
            "tags": ["jazz", "smooth", "background"],
        },
        {
            "genre": "Acoustic Pop",
            "bpm_range": (80, 100),
            "mood": "Neutral",
            "energy": "Medium",
            "valence": 0.60,
            "danceability": 0.55,
            "tags": ["acoustic", "mellow", "easy"],
        },
    ],
}


class MusicRecommender:
    """
    Emotion-driven music recommendation engine.

    Takes an emotion classification result and returns a ranked list
    of music track recommendations based on genre/mood/BPM profiles.

    Args:
        top_k: Number of recommendations to return per query.
        seed: Random seed for reproducibility.

    Example:
        >>> recommender = MusicRecommender(top_k=3)
        >>> tracks = recommender.recommend("joy", confidence=0.94)
        >>> for t in tracks:
        ...     print(t.genre, t.bpm)
    """

    def __init__(self, top_k: int = 3, seed: Optional[int] = 42) -> None:
        self.top_k = top_k
        self._rng = random.Random(seed)
        logger.info(f"MusicRecommender initialized (top_k={top_k})")

    def recommend(
        self, emotion: str, confidence: float = 1.0
    ) -> list[TrackRecommendation]:
        """
        Generate music recommendations based on a detected emotion.

        Args:
            emotion: Detected emotion label (e.g. 'joy', 'sadness').
            confidence: Classifier confidence score (0.0 to 1.0).

        Returns:
            List of TrackRecommendation objects ranked by relevance.
        """
        emotion = emotion.lower().strip()
        if emotion not in EMOTION_PROFILES:
            logger.warning(f"Unknown emotion '{emotion}', defaulting to 'neutral'.")
            emotion = "neutral"

        profiles = EMOTION_PROFILES[emotion]
        selected = self._rng.sample(profiles, k=min(self.top_k, len(profiles)))

        recommendations = []
        for profile in selected:
            bpm_lo, bpm_hi = profile["bpm_range"]
            bpm = self._rng.randint(bpm_lo, bpm_hi)
            rec = TrackRecommendation(
                genre=profile["genre"],
                bpm=bpm,
                mood=profile["mood"],
                energy=profile["energy"],
                valence=round(profile["valence"] * confidence, 4),
                danceability=round(profile["danceability"], 4),
                tags=profile["tags"],
            )
            recommendations.append(rec)

        logger.debug(
            f"Generated {len(recommendations)} recommendations for emotion='{emotion}'"
        )
        return recommendations

    def recommend_from_result(self, result) -> list[TrackRecommendation]:
        """
        Generate recommendations directly from an EmotionResult object.

        Args:
            result: EmotionResult from EmotionClassifier.predict().

        Returns:
            List of TrackRecommendation objects.
        """
        return self.recommend(result.emotion, result.confidence)

    @staticmethod
    def available_emotions() -> list[str]:
        """Return list of supported emotion labels."""
        return list(EMOTION_PROFILES.keys())

    def __repr__(self) -> str:
        return f"MusicRecommender(top_k={self.top_k})"
