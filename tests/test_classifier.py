"""
Unit tests for src/emotion_classifier.py  (EmotionClassifier).

Heavy model tests are skipped by default; use --run-model to include them.
Run with:  pytest tests/test_classifier.py -v
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from src.emotion_classifier import EmotionClassifier, EmotionResult


# ---------------------------------------------------------------------------
# Helpers / shared mocks
# ---------------------------------------------------------------------------


def _make_mock_model(emotion_label: str = "joy", num_labels: int = 7):
      """Build a mock DistilBert model that always predicts *emotion_label*."""
      label_list = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
      id2label = {i: lbl for i, lbl in enumerate(label_list)}
      label2id = {lbl: i for i, lbl in id2label.items()}

    predicted_id = label2id[emotion_label]
    # Create fake logits with highest value at predicted_id
    logits_data = [-10.0] * num_labels
    logits_data[predicted_id] = 10.0
    fake_logits = torch.tensor([logits_data])

    mock_model = MagicMock()
    mock_model.return_value.logits = fake_logits
    mock_model.config.id2label = id2label
    mock_model.config.label2id = label2id
    return mock_model, predicted_id


def _make_mock_tokenizer():
      """Build a mock tokenizer that returns dummy tensors."""
      mock_tok = MagicMock()
      mock_tok.return_value = {
          "input_ids": torch.zeros((1, 128), dtype=torch.long),
          "attention_mask": torch.ones((1, 128), dtype=torch.long),
      }
      return mock_tok


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def classifier_with_mocks():
      """EmotionClassifier with fully mocked model + tokenizer (no download)."""
      with (
                patch(
                              "src.emotion_classifier.DistilBertForSequenceClassification.from_pretrained"
                ) as mock_model_cls,
          patch(
                        "src.emotion_classifier.DistilBertTokenizerFast.from_pretrained"
          ) as mock_tok_cls,
          ):
                    mock_model, _ = _make_mock_model("joy")
                    mock_model_cls.return_value = mock_model
                    mock_tok_cls.return_value = _make_mock_tokenizer()

          clf = EmotionClassifier(device="cpu")
        yield clf


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------


class TestDeviceResolution:
      def test_explicit_device_cpu(self, classifier_with_mocks):
                clf = classifier_with_mocks
                clf.device = "cpu"
                assert clf.device == "cpu"

    def test_resolve_device_returns_string(self, classifier_with_mocks):
              assert isinstance(classifier_with_mocks.device, str)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


class TestPredict:
      def test_predict_returns_emotion_result(self, classifier_with_mocks):
                result = classifier_with_mocks.predict("I feel amazing!")
                assert isinstance(result, EmotionResult)

    def test_predict_result_has_correct_fields(self, classifier_with_mocks):
              result = classifier_with_mocks.predict("This is great")
              assert hasattr(result, "text")
              assert hasattr(result, "emotion")
              assert hasattr(result, "confidence")
              assert hasattr(result, "all_scores")

    def test_predict_preserves_original_text(self, classifier_with_mocks):
              text = "I feel absolutely wonderful today!"
              result = classifier_with_mocks.predict(text)
              assert result.text == text

    def test_emotion_is_string(self, classifier_with_mocks):
              result = classifier_with_mocks.predict("hello")
              assert isinstance(result.emotion, str)

    def test_confidence_between_0_and_1(self, classifier_with_mocks):
              result = classifier_with_mocks.predict("test input")
              assert 0.0 <= result.confidence <= 1.0

    def test_all_scores_is_dict(self, classifier_with_mocks):
              result = classifier_with_mocks.predict("test input")
              assert isinstance(result.all_scores, dict)

    def test_all_scores_sum_to_approx_1(self, classifier_with_mocks):
              result = classifier_with_mocks.predict("some text here")
              total = sum(result.all_scores.values())
              assert abs(total - 1.0) < 1e-4

    def test_empty_string_raises_value_error(self, classifier_with_mocks):
              with pytest.raises(ValueError):
                            classifier_with_mocks.predict("")

          def test_whitespace_only_raises_value_error(self, classifier_with_mocks):
                    with pytest.raises(ValueError):
                                  classifier_with_mocks.predict("   ")


# ---------------------------------------------------------------------------
# Batch prediction
# ---------------------------------------------------------------------------


class TestPredictBatch:
      def test_batch_returns_list(self, classifier_with_mocks):
                results = classifier_with_mocks.predict_batch(["hello", "world"])
                assert isinstance(results, list)

      def test_batch_length_matches_input(self, classifier_with_mocks):
                texts = ["one", "two", "three"]
                results = classifier_with_mocks.predict_batch(texts)
                assert len(results) == 3

      def test_batch_results_are_emotion_result(self, classifier_with_mocks):
                results = classifier_with_mocks.predict_batch(["a", "b"])
                assert all(isinstance(r, EmotionResult) for r in results)


# ---------------------------------------------------------------------------
# Labels property
# ---------------------------------------------------------------------------


class TestLabels:
      def test_labels_returns_list(self, classifier_with_mocks):
                assert isinstance(classifier_with_mocks.labels, list)

      def test_labels_are_strings(self, classifier_with_mocks):
                assert all(isinstance(l, str) for l in classifier_with_mocks.labels)


# ---------------------------------------------------------------------------
# EmotionResult dataclass
# ---------------------------------------------------------------------------


class TestEmotionResult:
      def test_repr_contains_emotion(self):
                er = EmotionResult(
                              text="test",
                              emotion="joy",
                              confidence=0.95,
                              all_scores={"joy": 0.95, "sadness": 0.05},
                )
                assert "joy" in repr(er)

      def test_repr_contains_confidence(self):
                er = EmotionResult(
                              text="test",
                              emotion="joy",
                              confidence=0.9412,
                              all_scores={},
                )
                assert "0.9412" in repr(er)


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------


def test_classifier_repr(classifier_with_mocks):
      r = repr(classifier_with_mocks)
      assert "EmotionClassifier" in r
