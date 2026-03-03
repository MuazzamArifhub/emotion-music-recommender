"""
Unit tests for src/music_recommender.py  (MusicRecommender).
Run with:  pytest tests/test_recommender.py -v
"""

from dataclasses import asdict

import pytest

from src.emotion_classifier import EmotionResult
from src.music_recommender import (
    EMOTION_PROFILES,
    MusicRecommender,
    TrackRecommendation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def recommender():
      """Default recommender with top_k=3 and fixed seed."""
      return MusicRecommender(top_k=3, seed=42)


@pytest.fixture
def joy_result():
      """Fake EmotionResult for 'joy'."""
      return EmotionResult(
          text="I just got promoted!",
          emotion="joy",
          confidence=0.94,
          all_scores={"joy": 0.94, "neutral": 0.06},
      )


# ---------------------------------------------------------------------------
# Basic recommendations
# ---------------------------------------------------------------------------


class TestRecommend:
      def test_returns_list(self, recommender):
                assert isinstance(recommender.recommend("joy"), list)

      def test_list_contains_track_recommendations(self, recommender):
                results = recommender.recommend("joy")
                assert all(isinstance(r, TrackRecommendation) for r in results)

      def test_top_k_limit_respected(self, recommender):
                results = recommender.recommend("joy")
                assert len(results) <= recommender.top_k

      def test_unknown_emotion_defaults_to_neutral(self, recommender):
                results = recommender.recommend("unknown_emotion_xyz")
                assert len(results) > 0  # neutral profile used

    def test_case_insensitive_emotion(self, recommender):
              results_lower = recommender.recommend("joy")
              results_upper = recommender.recommend("JOY")
              # Both should produce valid recommendations
              assert len(results_lower) > 0
              assert len(results_upper) > 0

    def test_all_supported_emotions(self, recommender):
              for emotion in EMOTION_PROFILES:
                            results = recommender.recommend(emotion)
                            assert len(results) > 0, f"No results for emotion: {emotion}"


# ---------------------------------------------------------------------------
# TrackRecommendation fields
# ---------------------------------------------------------------------------


class TestTrackRecommendationFields:
      def test_genre_is_string(self, recommender):
                rec = recommender.recommend("joy")[0]
                assert isinstance(rec.genre, str)

      def test_bpm_is_int(self, recommender):
                rec = recommender.recommend("joy")[0]
                assert isinstance(rec.bpm, int)

      def test_bpm_in_valid_range(self, recommender):
                rec = recommender.recommend("joy")[0]
                assert 40 <= rec.bpm <= 220

      def test_mood_is_string(self, recommender):
                rec = recommender.recommend("joy")[0]
                assert isinstance(rec.mood, str)

      def test_energy_is_string(self, recommender):
                rec = recommender.recommend("joy")[0]
                assert isinstance(rec.energy, str)

      def test_valence_between_0_and_1(self, recommender):
                rec = recommender.recommend("joy")[0]
                assert 0.0 <= rec.valence <= 1.0

      def test_danceability_between_0_and_1(self, recommender):
                rec = recommender.recommend("joy")[0]
                assert 0.0 <= rec.danceability <= 1.0

      def test_tags_is_list(self, recommender):
                rec = recommender.recommend("joy")[0]
                assert isinstance(rec.tags, list)

      def test_to_dict_returns_dict(self, recommender):
                rec = recommender.recommend("joy")[0]
                d = rec.to_dict()
                assert isinstance(d, dict)

      def test_to_dict_has_required_keys(self, recommender):
                rec = recommender.recommend("joy")[0]
                d = rec.to_dict()
                for key in ("genre", "bpm", "mood", "energy", "valence", "danceability", "tags"):
                              assert key in d


# ---------------------------------------------------------------------------
# recommend_from_result
# ---------------------------------------------------------------------------


class TestRecommendFromResult:
      def test_recommend_from_result_returns_list(self, recommender, joy_result):
                results = recommender.recommend_from_result(joy_result)
                assert isinstance(results, list)

      def test_recommend_from_result_non_empty(self, recommender, joy_result):
                results = recommender.recommend_from_result(joy_result)
                assert len(results) > 0

      def test_recommend_from_result_track_recommendations(self, recommender, joy_result):
                results = recommender.recommend_from_result(joy_result)
                assert all(isinstance(r, TrackRecommendation) for r in results)


# ---------------------------------------------------------------------------
# available_emotions
# ---------------------------------------------------------------------------


class TestAvailableEmotions:
      def test_returns_list(self):
                assert isinstance(MusicRecommender.available_emotions(), list)

      def test_contains_all_seven_emotions(self):
                emotions = MusicRecommender.available_emotions()
                expected = {"joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"}
                assert expected.issubset(set(emotions))


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
      def test_same_seed_same_results(self):
                r1 = MusicRecommender(top_k=3, seed=99)
                r2 = MusicRecommender(top_k=3, seed=99)
                recs1 = [rec.genre for rec in r1.recommend("sadness")]
                recs2 = [rec.genre for rec in r2.recommend("sadness")]
                assert recs1 == recs2


# ---------------------------------------------------------------------------
# Confidence scaling
# ---------------------------------------------------------------------------


class TestConfidenceScaling:
      def test_low_confidence_lowers_valence(self, recommender):
                recs_high = recommender.recommend("joy", confidence=1.0)
                recommender2 = MusicRecommender(top_k=3, seed=42)
                recs_low = recommender2.recommend("joy", confidence=0.1)
                # With same seed, same genre order; valence should be lower for low confidence
                for r_high, r_low in zip(recs_high, recs_low):
                              assert r_low.valence <= r_high.valence


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------


def test_repr(recommender):
      r = repr(recommender)
      assert "MusicRecommender" in r
