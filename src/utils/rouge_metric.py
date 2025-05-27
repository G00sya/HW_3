import re
import string
from collections import defaultdict
from typing import Dict, List, Tuple

from natasha import Doc, MorphVocab, NewsEmbedding, NewsMorphTagger, Segmenter


class RussianRouge:
    def __init__(self, ngram_sizes: Tuple[int, ...] = (1, 2), use_lemmatization: bool = True) -> None:
        """
        Initialize the Russian ROUGE scoring metric calculator.

        :param ngram_sizes: Tuple of n-gram sizes to calculate (1 for ROUGE-1, 2 for ROUGE-2, etc.)
        :param use_lemmatization: Whether to use lemmatization (recommended for Russian)
        """
        self.ngram_sizes = ngram_sizes
        self.use_lemmatization = use_lemmatization

        # Initialize Natasha components
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()
        self.emb = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(self.emb)

        # Russian stopwords (built-in to avoid NLTK dependency)
        self.stopwords = {
            "и",
            "в",
            "во",
            "не",
            "что",
            "он",
            "на",
            "я",
            "с",
            "со",
            "как",
            "а",
            "то",
            "все",
            "она",
            "так",
            "его",
            "но",
            "да",
            "ты",
            "к",
            "у",
            "же",
            "вы",
            "за",
            "бы",
            "по",
            "только",
            "ее",
            "мне",
            "было",
            "вот",
            "от",
            "меня",
            "еще",
            "нет",
            "о",
            "из",
            "ему",
            "теперь",
            "когда",
            "даже",
            "ну",
            "вдруг",
            "ли",
            "если",
            "уже",
            "или",
            "ни",
            "быть",
            "был",
            "него",
            "до",
            "вас",
            "нибудь",
            "опять",
            "уж",
            "вам",
            "ведь",
            "там",
            "потом",
            "себя",
            "ничего",
            "ей",
            "может",
            "они",
            "тут",
            "где",
            "есть",
            "надо",
            "ней",
            "для",
            "мы",
            "тебя",
            "их",
            "чем",
            "была",
            "сам",
            "чтоб",
            "без",
            "будто",
            "чего",
            "раз",
            "тоже",
            "себе",
            "под",
            "будет",
            "ж",
            "тогда",
            "кто",
            "этот",
            "того",
            "потому",
            "этого",
            "какой",
            "совсем",
            "ним",
            "здесь",
            "этом",
            "один",
            "почти",
            "мой",
            "тем",
            "чтобы",
            "нее",
            "сейчас",
            "были",
            "куда",
            "зачем",
            "всех",
            "никогда",
            "можно",
            "при",
            "наконец",
            "два",
            "об",
            "другой",
            "хоть",
            "после",
            "над",
            "больше",
            "тот",
            "через",
            "эти",
            "нас",
            "про",
            "всего",
            "них",
            "какая",
            "много",
            "разве",
            "три",
            "эту",
            "моя",
            "впрочем",
            "хорошо",
            "свою",
            "этой",
            "перед",
            "иногда",
            "лучше",
            "чуть",
            "том",
            "нельзя",
            "такой",
            "им",
            "более",
            "всегда",
            "конечно",
            "всю",
            "между",
        }
        self.punctuation = set(string.punctuation + "«»—…")

    def _preprocess_text(self, text: str) -> List[str]:
        """
        Tokenize and normalize text for ROUGE calculation.

        :param text: Input text to preprocess
        :return: List of processed tokens
        """
        # Normalize text (numbers to NUM, lowercase, remove punctuation)
        text = re.sub(r"\d+", "NUM", text)
        text = re.sub(r"[^\w\s]", " ", text.lower())

        # Process with Natasha pipeline
        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)

        tokens = []
        for token in doc.tokens:
            word = token.text.lower()
            if word in self.stopwords or word in self.punctuation:
                continue

            if self.use_lemmatization:
                token.lemmatize(self.morph_vocab)
                # Improve normalization for adjectives/verbs
                if token.lemma.endswith(("ый", "ой", "ий")):
                    tokens.append(token.lemma[:-2] + "ий")  # Normalize endings
                else:
                    tokens.append(token.lemma)
            else:
                tokens.append(word)

        return tokens

    def _get_ngrams(self, tokens: List[str], n: int) -> Dict[Tuple[str, ...], int]:
        """
        Extract n-grams with frequency counts.

        :param tokens: List of tokens to process
        :param n: Size of n-grams to extract
        :return: Dictionary of n-grams with their counts
        """
        ngrams = defaultdict(int)
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i : i + n])
            ngrams[ngram] += 1
        return ngrams

    def compute_score(self, candidate: str, reference: str) -> Dict[str, Dict[str, float]]:
        """
        Calculate ROUGE metrics between candidate and reference texts.

        :param candidate: Generated text to evaluate
        :param reference: Ground truth reference text
        :return: Dictionary with precision, recall and f1 for each n-gram size:
            {
                'rouge-1': {'precision': 0.5, 'recall': 0.3, 'f1': 0.4},
                'rouge-2': {'precision': 0.2, 'recall': 0.1, 'f1': 0.15}
            }
        """
        # Preprocess texts
        cand_tokens = self._preprocess_text(candidate)
        ref_tokens = self._preprocess_text(reference)

        scores = {}

        for n in self.ngram_sizes:
            # Extract n-grams
            cand_ngrams = self._get_ngrams(cand_tokens, n)
            ref_ngrams = self._get_ngrams(ref_tokens, n)

            # Calculate overlaps
            overlap = sum(min(cand_ngrams[ngram], ref_ngrams[ngram]) for ngram in set(cand_ngrams) & set(ref_ngrams))

            # Compute metrics
            precision = overlap / sum(cand_ngrams.values()) if cand_ngrams else 0
            recall = overlap / sum(ref_ngrams.values()) if ref_ngrams else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0

            scores[f"rouge-{n}"] = {"precision": precision, "recall": recall, "f1": f1}

        return scores
