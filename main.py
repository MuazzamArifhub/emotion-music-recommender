"""
main.py - CLI entry point for Emotion Music Recommender.

Usage:
    python main.py --text "I just got promoted and I feel amazing!"
    python main.py --text "I feel so sad today" --top-k 2
        python main.py --text "Everything is fine" --device cpu --no-pretty
        """

from __future__ import annotations

import argparse
import json
import logging
import sys

from src.emotion_classifier import EmotionClassifier
from src.music_recommender import MusicRecommender
from src.preprocessor import TextPreprocessor


def setup_logging(level: str = "WARNING") -> None:
      """Configure basic logging."""
      logging.basicConfig(
          level=getattr(logging, level.upper(), logging.WARNING),
          format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
          datefmt="%Y-%m-%d %H:%M:%S",
      )


def build_parser() -> argparse.ArgumentParser:
      parser = argparse.ArgumentParser(
                prog="emotion-music-recommender",
                description=(
                              "Detect emotion from text and receive music recommendations "
                              "powered by a fine-tuned DistilBERT transformer model."
                ),
      )
      parser.add_argument(
          "--text",
          type=str,
          required=True,
          help="Input text to analyse (e.g. 'I feel amazing today!')",
      )
      parser.add_argument(
          "--top-k",
          type=int,
          default=3,
          metavar="K",
          help="Number of music recommendations to return (default: 3)",
      )
      parser.add_argument(
          "--device",
          type=str,
          default=None,
          choices=["cpu", "cuda", "mps"],
          help="Compute device for the model (default: auto-detect)",
      )
      parser.add_argument(
          "--no-pretty",
          action="store_true",
          help="Output compact JSON instead of pretty-printed",
      )
      parser.add_argument(
          "--log-level",
          type=str,
          default="WARNING",
          choices=["DEBUG", "INFO", "WARNING", "ERROR"],
          help="Logging verbosity (default: WARNING)",
      )
      return parser


def run(
      text: str,
      top_k: int = 3,
      device: str | None = None,
      pretty: bool = True,
) -> dict:
      """
          Execute the full emotion -> music pipeline.

              Args:
                      text: Raw input text.
                              top_k: Number of recommendations to return.
                                      device: Torch device string or None for auto-detection.
                                              pretty: Whether to pretty-print the JSON output.

                                                  Returns:
                                                          dict with keys: ``emotion``, ``confidence``, ``recommendations``.
                                                              """
      # 1. Preprocess
      preprocessor = TextPreprocessor()
      preprocessed = preprocessor.process(text)

    # 2. Classify emotion
      classifier = EmotionClassifier(device=device)
      emotion_result = classifier.predict(preprocessed.cleaned)

    # 3. Recommend music
      recommender = MusicRecommender(top_k=top_k)
      tracks = recommender.recommend_from_result(emotion_result)

    # 4. Build output payload
      output = {
          "emotion": emotion_result.emotion,
          "confidence": round(emotion_result.confidence, 4),
          "recommendations": [t.to_dict() for t in tracks],
      }

    # 5. Print
      indent = 2 if pretty else None
      print(json.dumps(output, indent=indent))
      return output


def main() -> None:
      parser = build_parser()
      args = parser.parse_args()

    setup_logging(args.log_level)

    try:
              run(
                            text=args.text,
                            top_k=args.top_k,
                            device=args.device,
                            pretty=not args.no_pretty,
              )
except ValueError as exc:
          print(f"Error: {exc}", file=sys.stderr)
          sys.exit(1)
except Exception as exc:  # pragma: no cover
          print(f"Unexpected error: {exc}", file=sys.stderr)
          sys.exit(2)


if __name__ == "__main__":
      main()
