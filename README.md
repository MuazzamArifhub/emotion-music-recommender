# Emotion Music Recommender
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange?logo=pytorch)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)](https://huggingface.co)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> Detect emotions from natural language text and receive intelligent music recommendations powered by a fine-tuned DistilBERT transformer model.

## Overview

**Emotion Music Recommender** is an end-to-end ML pipeline that:

1. Accepts raw text input (journal entries, messages, mood descriptions)
2. Classifies the emotion using a fine-tuned DistilBERT model (7 emotion classes)
3. Maps the detected emotion to music genre, tempo, and mood attributes
4. Returns curated track recommendations with confidence scores

## Architecture

```
Input Text
    |
[Preprocessor]       - Tokenization, cleaning, normalization
    |
[EmotionClassifier]  - DistilBERT fine-tuned on GoEmotions
    |
[MusicRecommender]   - Emotion to Genre/BPM/Mood mapping
    |
Track Recommendations + Confidence Scores
```

## Supported Emotions

| Emotion | Genre | BPM | Mood |
|---------|-------|-----|------|
| Joy | Pop / Dance | 120-140 | Uplifting |
| Sadness | Indie / Blues | 60-80 | Melancholic |
| Anger | Metal / Rock | 140-180 | Intense |
| Fear | Ambient / Classical | 50-70 | Calm |
| Surprise | Electronic / Funk | 110-130 | Energetic |
| Disgust | Punk / Grunge | 130-160 | Raw |
| Neutral | Lo-fi / Jazz | 70-90 | Relaxed |

## Project Structure

```
emotion-music-recommender/
├── src/
│   ├── __init__.py
│   ├── preprocessor.py
│   ├── emotion_classifier.py
│   └── music_recommender.py
├── config/
│   └── config.yaml
├── tests/
│   ├── test_preprocessor.py
│   ├── test_classifier.py
│   └── test_recommender.py
├── main.py
├── demo.py
├── evaluate.py
└── requirements.txt
```

## Quick Start

```bash
git clone https://github.com/MuazzamArifhub/emotion-music-recommender.git
cd emotion-music-recommender
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python demo.py
```

## CLI Usage

```bash
python main.py --text "I just got promoted and I feel amazing!"
```

Output:
```json
{
  "emotion": "joy",
  "confidence": 0.94,
  "recommendations": [
    {"genre": "Pop", "bpm": 128, "mood": "Uplifting", "energy": "High"},
    {"genre": "Dance", "bpm": 132, "mood": "Euphoric", "energy": "Very High"}
  ]
}
```

## Model Details

- **Base**: distilbert-base-uncased
- **Dataset**: GoEmotions (58k examples, 7 emotion classes)
- **Training**: 3 epochs, AdamW, lr=2e-5
- **Accuracy**: ~89% on validation set

## Running Tests

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

## License

[MIT](LICENSE) - MuazzamArifhub 2026Detect emotions from text and get smart music recommendations using DistilBERT.
