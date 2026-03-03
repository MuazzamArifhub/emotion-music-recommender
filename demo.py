"""
demo.py - Interactive demonstration of the Emotion Music Recommender.

Runs through a set of example inputs covering all 7 supported emotions
and pretty-prints the full pipeline output for each one.

Usage:
    python demo.py
    """

from __future__ import annotations

import json
import logging

from src.emotion_classifier import EmotionClassifier
from src.music_recommender import MusicRecommender
from src.preprocessor import TextPreprocessor

# Suppress transformer download logs during demo
logging.basicConfig(level=logging.ERROR)

# ---------------------------------------------------------------------------
# Demo inputs - one per supported emotion
# ---------------------------------------------------------------------------
DEMO_INPUTS = [
      "I just got promoted and I feel absolutely amazing!",
      "I've been crying all day, everything feels so hopeless.",
      "I'm so furious right now, I can't believe they did that!",
      "I'm terrified of what might happen next, I can't stop shaking.",
      "Wow, I had no idea that was going to happen - totally unexpected!",
      "This is disgusting, I can't stand being around these people.",
      "Just another regular Tuesday. Nothing special going on.",
]

SEPARATOR = "=" * 60


def run_demo() -> None:
      print(SEPARATOR)
      print("  Emotion Music Recommender - Live Demo")
      print(SEPARATOR)
      print("Loading model (this may take a moment on first run)...\n")

    # Initialise pipeline components once
      preprocessor = TextPreprocessor()
      classifier = EmotionClassifier()
      recommender = MusicRecommender(top_k=3)

    for i, text in enumerate(DEMO_INPUTS, start=1):
              print(f"\n[{i}/{len(DEMO_INPUTS)}] Input: \"{text}\"")
              print("-" * 60)

        # Step 1: Preprocess
              preprocessed = preprocessor.process(text)

        # Step 2: Classify
              result = classifier.predict(preprocessed.cleaned)

        # Step 3: Recommend
              tracks = recommender.recommend_from_result(result)

        # Step 4: Display
              print(f"  Detected emotion : {result.emotion.upper()}")
              print(f"  Confidence       : {result.confidence:.4f}")
              print(f"  Recommendations  :")
              for j, track in enumerate(tracks, start=1):
                            print(
                                              f"    {j}. {track.genre:<25} | BPM {track.bpm:>3} "
                                              f"| {track.mood:<12} | Energy: {track.energy}"
                            )

          print(f"\n{SEPARATOR}")
    print("  Demo complete.")
    print(SEPARATOR)


if __name__ == "__main__":
      run_demo()
