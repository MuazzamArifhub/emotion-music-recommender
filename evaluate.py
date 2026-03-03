"""
evaluate.py - Offline evaluation script for the EmotionClassifier.

Measures accuracy, per-class precision/recall/F1, and confusion matrix
on a labelled evaluation dataset (CSV format).

Expected CSV columns:
    text  - raw input text
        label - ground-truth emotion label

        Usage:
            python evaluate.py --data eval_data.csv
                python evaluate.py --data eval_data.csv --device cpu --output results.json
                """

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s | %(levelname)s | %(message)s",
      datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Built-in mini eval set (used when --data is not provided)
# ---------------------------------------------------------------------------
BUILTIN_EVAL_SET: list[dict[str, str]] = [
      {"text": "I feel absolutely wonderful and full of joy!", "label": "joy"},
      {"text": "This is the happiest day of my life!", "label": "joy"},
      {"text": "I am devastated, nothing will ever be okay.", "label": "sadness"},
      {"text": "I've been crying for hours, I feel so alone.", "label": "sadness"},
      {"text": "I am furious! How could they do this to me?", "label": "anger"},
      {"text": "This makes my blood boil, I'm absolutely livid.", "label": "anger"},
      {"text": "I'm scared of what comes next. I can't breathe.", "label": "fear"},
      {"text": "Everything is frightening me right now.", "label": "fear"},
      {"text": "I can't believe that just happened - wow!", "label": "surprise"},
      {"text": "That was totally unexpected and shocking.", "label": "surprise"},
      {"text": "This is absolutely revolting and repulsive.", "label": "disgust"},
      {"text": "I find this deeply offensive and disgusting.", "label": "disgust"},
      {"text": "Just another quiet day, nothing much going on.", "label": "neutral"},
      {"text": "Everything is fine, nothing to report.", "label": "neutral"},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_csv(path: str) -> list[dict[str, str]]:
      """Load a CSV file into a list of {text, label} dicts."""
      import csv
      records = []
      with open(path, newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                              if "text" not in row or "label" not in row:
                                                raise ValueError("CSV must contain 'text' and 'label' columns.")
                                            records.append({"text": row["text"], "label": row["label"].lower().strip()})
                      return records


def compute_metrics(
      y_true: list[str], y_pred: list[str], labels: list[str]
) -> dict:
      """Compute accuracy, per-class precision/recall/F1, and confusion matrix."""
      n = len(y_true)
      assert n == len(y_pred), "Length mismatch between y_true and y_pred."

    # Accuracy
      correct = sum(t == p for t, p in zip(y_true, y_pred))
      accuracy = correct / n if n > 0 else 0.0

    # Per-class counts
      label_set = sorted(set(labels))
      tp: dict[str, int] = {l: 0 for l in label_set}
      fp: dict[str, int] = {l: 0 for l in label_set}
      fn: dict[str, int] = {l: 0 for l in label_set}

    for t, p in zip(y_true, y_pred):
              if t == p:
                            tp[t] = tp.get(t, 0) + 1
else:
              fp[p] = fp.get(p, 0) + 1
              fn[t] = fn.get(t, 0) + 1

    per_class: dict[str, dict] = {}
    for lbl in label_set:
              prec_denom = tp[lbl] + fp.get(lbl, 0)
              rec_denom = tp[lbl] + fn.get(lbl, 0)
              precision = tp[lbl] / prec_denom if prec_denom else 0.0
              recall = tp[lbl] / rec_denom if rec_denom else 0.0
              f1_denom = precision + recall
              f1 = 2 * precision * recall / f1_denom if f1_denom else 0.0
              per_class[lbl] = {
                  "precision": round(precision, 4),
                  "recall": round(recall, 4),
                  "f1": round(f1, 4),
                  "support": rec_denom,
              }

    # Confusion matrix (label_set rows = true, cols = pred)
    cm: dict[str, dict[str, int]] = {l: {l2: 0 for l2 in label_set} for l in label_set}
    for t, p in zip(y_true, y_pred):
              if t in cm and p in cm[t]:
                            cm[t][p] += 1

          # Macro-average F1
          macro_f1 = sum(v["f1"] for v in per_class.values()) / len(per_class) if per_class else 0.0

    return {
              "accuracy": round(accuracy, 4),
              "macro_f1": round(macro_f1, 4),
              "total_samples": n,
              "correct": correct,
              "per_class": per_class,
              "confusion_matrix": cm,
    }


def print_report(metrics: dict) -> None:
      """Print a formatted evaluation report to stdout."""
      print("\n" + "=" * 60)
      print("  EVALUATION REPORT")
      print("=" * 60)
      print(f"  Accuracy   : {metrics['accuracy']:.4f}  ({metrics['correct']}/{metrics['total_samples']})")
      print(f"  Macro F1   : {metrics['macro_f1']:.4f}")
      print("\n  Per-class metrics:")
      print(f"  {'Label':<12} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>9}")
      print("  " + "-" * 52)
      for lbl, vals in sorted(metrics["per_class"].items()):
                print(
                              f"  {lbl:<12} {vals['precision']:>10.4f} {vals['recall']:>8.4f} "
                              f"{vals['f1']:>8.4f} {vals['support']:>9}"
                )
            print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
      parser = argparse.ArgumentParser(
          prog="evaluate",
          description="Evaluate the EmotionClassifier on a labelled dataset.",
)
    parser.add_argument(
              "--data",
              type=str,
              default=None,
              help="Path to CSV file with 'text' and 'label' columns. "
                   "Uses built-in mini eval set if omitted.",
    )
    parser.add_argument(
              "--device",
              type=str,
              default=None,
              choices=["cpu", "cuda", "mps"],
              help="Torch device (default: auto-detect).",
    )
    parser.add_argument(
              "--output",
              type=str,
              default=None,
              help="Optional path to save JSON results.",
    )
    parser.add_argument(
              "--batch-size",
              type=int,
              default=1,
              metavar="N",
              help="Inference batch size (default: 1).",
    )
    return parser


def main() -> None:
      parser = build_parser()
    args = parser.parse_args()

    # Load data
    if args.data:
              logger.info(f"Loading evaluation data from '{args.data}'")
              try:
                            records = load_csv(args.data)
except FileNotFoundError:
            print(f"Error: file not found: {args.data}", file=sys.stderr)
            sys.exit(1)
except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
else:
        logger.info("No --data provided, using built-in evaluation set.")
          records = BUILTIN_EVAL_SET

    logger.info(f"Loaded {len(records)} evaluation samples.")

    # Import here so that the module is importable without heavy deps
    from src.emotion_classifier import EmotionClassifier
    from src.preprocessor import TextPreprocessor

    preprocessor = TextPreprocessor()
    classifier = EmotionClassifier(device=args.device)

    # Run inference
    y_true: list[str] = []
    y_pred: list[str] = []

    logger.info("Running inference...")
    for i, record in enumerate(records):
              try:
                            cleaned = preprocessor.process(record["text"]).cleaned
                            result = classifier.predict(cleaned)
                            y_true.append(record["label"])
                            y_pred.append(result.emotion)
except Exception as exc:
            logger.warning(f"Skipping sample {i}: {exc}")

    # Compute metrics
    all_labels = list(
              set(y_true) | set(y_pred) | {"joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"}
    )
    metrics = compute_metrics(y_true, y_pred, all_labels)

    # Print report
    print_report(metrics)

    # Optionally save JSON
    if args.output:
              out_path = Path(args.output)
              out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
              logger.info(f"Results saved to '{args.output}'")


if __name__ == "__main__":
      main()
