from collections import defaultdict, Counter
import random
import math
# Trigram Language Model Implementation
class TrigramLM:
    def __init__(self, delta=0.01):
        self.delta = delta
        self.histogram = defaultdict(Counter)
        self.symbols = set()
        self.ready = False

    def ingest_file(self, path_to_data):
        """
        Populate trigram counts using padded lines from training file.
        """
        with open(path_to_data, 'r', encoding='utf-8') as handle:
            for sequence in handle:
                chain = '##' + sequence.strip()
                for idx in range(len(chain) - 2):
                    prefix = chain[idx:idx+2]
                    suffix = chain[idx+2]
                    self.histogram[prefix][suffix] += 1
                    self.symbols.update(prefix + suffix)

        self.symbols = sorted(self.symbols)
        self.symbol_space = len(self.symbols)
        self.ready = True

    def prob(self, prefix, letter):
        """
        Smoothed conditional likelihood estimate.
        """
        possible_vals = self.histogram[prefix]
        volume = sum(possible_vals.values())
        return (possible_vals[letter] + self.delta) / (volume + self.delta * self.symbol_space)

    def synth(self, limit=100):
        """
        Sample characters until reaching max length.
        """
        if not self.ready:
            raise RuntimeError("Model not initialized.")

        trail = ['#', '#']
        while len(trail) < limit + 2:
            key = ''.join(trail[-2:])
            choices = list(self.symbols)
            weights = [self.prob(key, z) for z in choices]
            chosen = random.choices(choices, weights=weights, k=1)[0]
            trail.append(chosen)

        return ''.join(trail[2:])

    def perplexity_from_string(self, sentence):
        """
        Compute perplexity for a single input string.
        """
        if not self.ready:
            raise RuntimeError("Model not initialized.")

        sentence = '##' + sentence.strip()
        bits = 0.0
        chars = 0
        for k in range(len(sentence) - 2):
            context = sentence[k:k+2]
            next_char = sentence[k+2]
            prob = self.prob(context, next_char)
            bits += math.log2(prob)
            chars += 1

        return 2 ** (-bits / chars) if chars > 0 else float('inf')


    def perplexity(self, input_file):
        """
        Compute perplexity of held-out file.
        """
        if not self.ready:
            raise RuntimeError("Model not initialized.")

        bits = 0.0
        chars = 0

        with open(input_file, 'r', encoding='utf-8') as fstream:
            for row in fstream:
                strand = '##' + row.strip()
                for k in range(len(strand) - 2):
                    past = strand[k:k+2]
                    next_ch = strand[k+2]
                    likelihood = self.prob(past, next_ch)
                    bits += math.log2(likelihood)
                    chars += 1

        return 2 ** (-bits / chars)
