import pytest

from src.utils.rouge_metric import RussianRouge


@pytest.fixture
def rouge():
    """Fixture providing initialized RussianRouge instance for tests."""
    return RussianRouge(ngram_sizes=(1, 2), use_lemmatization=True)


def test_preprocess_text(rouge):
    """Test text preprocessing (tokenization, lemmatization and cleaning)."""
    text = "Кошка, лежит на диване!"
    result = rouge._preprocess_text(text)
    assert result == ["кошка", "лежать", "диван"]  # Verify lemmatization and cleaning


def test_get_ngrams(rouge):
    """Test n-gram extraction functionality."""
    tokens = ["кошка", "лежать", "диван"]
    bigrams = rouge._get_ngrams(tokens, 2)
    assert bigrams == {("кошка", "лежать"): 1, ("лежать", "диван"): 1}


def test_compute_score_perfect_match(rouge):
    """Test perfect match between candidate and reference."""
    candidate = "Кошка лежит на диване"
    reference = "Кошка лежит на диване"
    scores = rouge.compute_score(candidate, reference)

    for n in [1, 2]:
        assert scores[f"rouge-{n}"]["precision"] == 1.0
        assert scores[f"rouge-{n}"]["recall"] == 1.0
        assert scores[f"rouge-{n}"]["f1"] == 1.0


def test_compute_score_partial_match(rouge):
    """Test partial match with different but similar texts."""
    candidate = "Большая кошка спит на ковре"
    reference = "Кошка лежит на диване"
    scores = rouge.compute_score(candidate, reference)

    # Natasha might produce slightly different lemmas, so we use wider threshold
    assert scores["rouge-1"]["f1"] > 0.25  # More lenient threshold
    assert scores["rouge-2"]["f1"] == 0.0  # No matching bigrams expected


def test_compute_score_no_match(rouge):
    """Test complete mismatch between unrelated texts."""
    candidate = "Собака бегает во дворе"
    reference = "Кошка лежит на диване"
    scores = rouge.compute_score(candidate, reference)

    for n in [1, 2]:
        assert scores[f"rouge-{n}"]["f1"] == 0.0


def test_without_lemmatization():
    """Test operation with lemmatization disabled."""
    rouge = RussianRouge(use_lemmatization=False)
    candidate = "Кошка лежит на диванах"
    reference = "Кошки лежат на диване"
    scores = rouge.compute_score(candidate, reference)

    # Without lemmatization we expect fewer matches
    assert scores["rouge-1"]["f1"] < 0.3


def test_empty_input(rouge):
    """Test edge case with empty input strings."""
    scores = rouge.compute_score("", "")
    for n in [1, 2]:
        assert scores[f"rouge-{n}"]["precision"] == 0
        assert scores[f"rouge-{n}"]["recall"] == 0
        assert scores[f"rouge-{n}"]["f1"] == 0


def test_stopwords_removal(rouge):
    """Test proper filtering of stopwords."""
    candidate = "И вот тогда я пошел домой"
    reference = "Он ушел домой"
    scores = rouge.compute_score(candidate, reference)

    # Main match should be on the word "домой"
    assert scores["rouge-1"]["f1"] > 0.4


def test_punctuation_removal(rouge):
    """Test proper handling of punctuation."""
    candidate = "Кошка (очень красивая!) лежит..."
    reference = "Кошка лежит на диване"
    scores = rouge.compute_score(candidate, reference)

    # Punctuation should not affect the matching
    assert scores["rouge-1"]["f1"] > 0.4
