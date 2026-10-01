"""Adapted from Module 2 Practical 2: tokenize, count, then sample bigrams."""

from collections import Counter, defaultdict
import random
import re


class BigramModel:
    def __init__(self, corpus: list[str]):
        self.transitions: dict[str, Counter] = defaultdict(Counter)
        for text in corpus:
            words = re.findall(r"\b\w+\b", text.lower())
            for current, following in zip(words, words[1:]):
                self.transitions[current][following] += 1

    def generate_text(self, start_word: str, length: int) -> str:
        words = [start_word.lower()]
        for _ in range(length - 1):
            counts = self.transitions.get(words[-1])
            if not counts:
                break
            # Relative counts give the same sampling weights as probabilities.
            following = random.choices(list(counts), weights=list(counts.values()))[0]
            words.append(following)
        return " ".join(words)
