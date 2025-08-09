import os
from collections import Counter, defaultdict
from itertools import combinations
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class FlatBytePairEncoder:
    def __init__(self):
        self.vocab = Counter()
        self.merges = []

    def preprocess(self, lines):
        """Flatten corpus into a single character list, marking word boundaries."""
        tokens = []
        for line in lines:
            for word in line.strip().split():
                tokens.extend(list(word) + ['</w>']) 
        
        return tokens

    def get_stats(self, tokens):
        """Count adjacent pair frequencies."""
        pairs = defaultdict(int)
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pairs[pair] += 1
        return pairs

    def merge_pair(self, tokens, pair_to_merge):
        """Merge a pair of symbols in the token stream."""
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair_to_merge[0] and tokens[i + 1] == pair_to_merge[1]:
                new_tokens.append(tokens[i] + tokens[i + 1])
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens

    def learn_bpe(self, lines, num_merges=100):
        """
    Learns BPE merges from a list of input lines using character pair frequencies.

    Returns:
        List of learned merge pairs.
    """
        tokens = self.preprocess(lines)
        for _ in range(num_merges):
            stats = self.get_stats(tokens)
            if not stats:
                break
            best_pair = max(stats, key=stats.get)
            self.merges.append(best_pair)
            tokens = self.merge_pair(tokens, best_pair)
        return self.merges

    def get_final_vocab(self):
        """
    Returns the final set of subword units after all BPE merges.
    """
        return set(''.join(pair) for pair in self.merges)


def run_bpe_analysis(data_dir='data/normalised', langs=['en', 'af', 'nl', 'xh', 'zu'], num_merges=100):
    """
    Runs BPE on multiple language corpora, computes pairwise vocab overlaps, and saves results + heatmap.
    """
    bpe_results = {}
    merge_table = {}

    for lang in langs:
        file_path = os.path.join(data_dir, f'train.{lang}.txt')
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        bpe = FlatBytePairEncoder()
        merges = bpe.learn_bpe(lines, num_merges=num_merges)
        final_vocab = bpe.get_final_vocab()

        bpe_results[lang] = final_vocab
        merge_table[lang] = merges[:10]

    with open('bpe_top10_merges.txt', 'w', encoding='utf-8') as fout:
        for lang in langs:
            fout.write(f'First 10 merges for {lang}:\n')
            for i, pair in enumerate(merge_table[lang]):
                fout.write(f'{i+1:2}: {pair[0]} + {pair[1]} -> {pair[0]+pair[1]}\n')
            fout.write('\n')

    overlap_matrix = pd.DataFrame(index=langs, columns=langs, dtype=float)
    for lang1, lang2 in combinations(langs, 2):
        v1, v2 = bpe_results[lang1], bpe_results[lang2]
        overlap = len(v1 & v2)
        total = len(v1 | v2)
        overlap_pct = 100 * overlap / total if total > 0 else 0
        overlap_matrix.loc[lang1, lang2] = overlap_pct
        overlap_matrix.loc[lang2, lang1] = overlap_pct
        print(f"  {lang1}-{lang2}: {overlap_pct:.2f}% overlap")

    for lang in langs:
        overlap_matrix.loc[lang, lang] = 100.0

    overlap_matrix.to_csv('bpe_overlap_matrix.csv')
    print("\n✅ BPE Overlap Matrix saved to 'bpe_overlap_matrix.csv'")
    print("\n📊 BPE Overlap Matrix (% of shared subwords):")
    print(overlap_matrix.round(2))
    plt.figure(figsize=(8, 6))
    sns.heatmap(overlap_matrix.astype(float), annot=True, fmt=".2f", cmap="Blues", cbar=True)
    plt.title("BPE Vocabulary Overlap (%)")
    plt.tight_layout()
    plt.savefig("bpe_overlap_heatmap.png", dpi=300)
    print("\n BPE Overlap Heatmap saved to 'bpe_overlap_heatmap.png'")


