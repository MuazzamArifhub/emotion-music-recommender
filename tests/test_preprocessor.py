"""
Unit tests for src/preprocessor.py  (TextPreprocessor).
Run with:  pytest tests/test_preprocessor.py -v
"""

import pytest

from src.preprocessor import PreprocessedText, TextPreprocessor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def preprocessor():
      """Default preprocessor with all options enabled."""
      return TextPreprocessor(
          lowercase=True,
          expand_contractions=True,
          remove_urls=True,
          remove_punctuation=False,
          max_words=None,
      )


@pytest.fixture
def strict_preprocessor():
      """Preprocessor with punctuation removal and word limit."""
      return TextPreprocessor(
          lowercase=True,
          expand_contractions=True,
          remove_urls=True,
          remove_punctuation=True,
          max_words=10,
      )


# ---------------------------------------------------------------------------
# Basic processing
# ---------------------------------------------------------------------------


class TestBasicProcessing:
      def test_returns_preprocessed_text_instance(self, preprocessor):
                result = preprocessor.process("Hello world")
                assert isinstance(result, PreprocessedText)

      def test_original_text_preserved(self, preprocessor):
                text = "Hello World!"
                result = preprocessor.process(text)
                assert result.original == text

      def test_lowercase(self, preprocessor):
                result = preprocessor.process("Hello WORLD")
                assert result.cleaned == "hello world"

      def test_word_count(self, preprocessor):
                result = preprocessor.process("one two three four")
                assert result.word_count == 4

      def test_char_count(self, preprocessor):
                result = preprocessor.process("hi")
                assert result.char_count == 2

      def test_was_truncated_false_by_default(self, preprocessor):
                result = preprocessor.process("short text")
                assert result.was_truncated is False


# ---------------------------------------------------------------------------
# Contraction expansion
# ---------------------------------------------------------------------------


class TestContractionExpansion:
      def test_cant_expanded(self, preprocessor):
                result = preprocessor.process("I can't do this")
                assert "cannot" in result.cleaned

      def test_dont_expanded(self, preprocessor):
                result = preprocessor.process("I don't know")
                assert "do not" in result.cleaned

      def test_wont_expanded(self, preprocessor):
                result = preprocessor.process("I won't go")
                assert "will not" in result.cleaned

      def test_its_expanded(self, preprocessor):
                result = preprocessor.process("It's fine")
                assert "it is" in result.cleaned

      def test_no_expansion_when_disabled(self):
                p = TextPreprocessor(expand_contractions=False)
                result = p.process("I can't believe it")
                assert "can't" in result.cleaned


# ---------------------------------------------------------------------------
# URL removal
# ---------------------------------------------------------------------------


class TestURLRemoval:
      def test_http_url_removed(self, preprocessor):
                result = preprocessor.process("Check out https://example.com today")
                assert "https" not in result.cleaned

      def test_www_url_removed(self, preprocessor):
                result = preprocessor.process("Visit www.example.com for info")
                assert "www" not in result.cleaned

      def test_no_removal_when_disabled(self):
                p = TextPreprocessor(remove_urls=False)
                result = p.process("See https://example.com")
                assert "https" in result.cleaned


# ---------------------------------------------------------------------------
# Punctuation removal
# ---------------------------------------------------------------------------


class TestPunctuationRemoval:
      def test_punctuation_removed(self):
                p = TextPreprocessor(remove_punctuation=True)
                result = p.process("Hello, world!")
                assert "," not in result.cleaned
                assert "!" not in result.cleaned

      def test_punctuation_preserved_by_default(self, preprocessor):
                result = preprocessor.process("Hello, world!")
                assert "!" in result.cleaned


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


class TestTruncation:
      def test_truncates_to_max_words(self):
                p = TextPreprocessor(max_words=3)
                result = p.process("one two three four five")
                assert result.word_count == 3
                assert result.was_truncated is True

      def test_no_truncation_when_within_limit(self):
                p = TextPreprocessor(max_words=10)
                result = p.process("only five words here")
                assert result.was_truncated is False


# ---------------------------------------------------------------------------
# Edge cases & error handling
# ---------------------------------------------------------------------------


class TestEdgeCases:
      def test_empty_string_raises_value_error(self, preprocessor):
                with pytest.raises(ValueError):
                              preprocessor.process("")

            def test_whitespace_only_raises_value_error(self, preprocessor):
                      with pytest.raises(ValueError):
                                    preprocessor.process("   ")

                  def test_extra_whitespace_collapsed(self, preprocessor):
                            result = preprocessor.process("too   many    spaces")
                            assert "  " not in result.cleaned

    def test_newlines_collapsed(self, preprocessor):
              result = preprocessor.process("line one\nline two")
              assert "\n" not in result.cleaned


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------


class TestBatchProcessing:
      def test_batch_returns_list(self, preprocessor):
                results = preprocessor.process_batch(["hello", "world"])
                assert isinstance(results, list)
                assert len(results) == 2

    def test_batch_results_are_preprocessed_text(self, preprocessor):
              results = preprocessor.process_batch(["hello", "world"])
              assert all(isinstance(r, PreprocessedText) for r in results)


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------


def test_repr(preprocessor):
      r = repr(preprocessor)
    assert "TextPreprocessor" in r
