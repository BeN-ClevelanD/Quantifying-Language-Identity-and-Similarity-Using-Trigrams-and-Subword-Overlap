import os
from triModel import TrigramLM
import pandas as pd
#This script evaluates different alpha values for smoothing in a trigram language model and saves the results to a CSV file.
TRAIN_DIR = 'data/normalised'
VAL_DIR = 'data'
LANGS = ['en', 'af', 'nl', 'xh', 'zu']
ALPHAS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
def evaluate_alpha(alpha):
    """
    Train trigram models with a given alpha and return validation perplexities.
    """
    results = {}
    for lang in LANGS:
        model = TrigramLM(delta=alpha)
        train_file = os.path.join(TRAIN_DIR, f'train.{lang}.txt')
        val_file = os.path.join(VAL_DIR, f'val.{lang}.txt')
        model.ingest_file(train_file)
        perplexity = model.perplexity(val_file)
        results[lang] = round(perplexity, 2)
    return results

def main():
    print("Evaluating different alpha values...\n")
    all_results = {}
    for alpha in ALPHAS:
        print(f"Alpha = {alpha}")
        result = evaluate_alpha(alpha)
        all_results[alpha] = result
        for lang, perp in result.items():
            print(f"  {lang.upper()} → {perp}")
        print()
    df = pd.DataFrame(all_results).T
    df.index.name = 'Alpha'
    df.to_csv('alpha_tuning_results.csv')
    print("✅ Results saved to 'alpha_tuning_results.csv'")
    print("\n📊 Validation Perplexity per Language:")
    print(df.round(2))

if __name__ == '__main__':
    main()
