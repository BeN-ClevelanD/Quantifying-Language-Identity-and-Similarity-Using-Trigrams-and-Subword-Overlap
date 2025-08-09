import os
from text_normaliser import normalise_text_file
import os
from triModel import TrigramLM 
import pprint
import pandas as pd
import matplotlib.pyplot as plotter
import seaborn as sea
from collections import namedtuple
import subprocess
from bpeimplement import run_bpe_analysis
import csv

#This script is the main and central program that runs all related components for all tasks in this project. See read me for explanation of components.
LANGUAGE_CODES = ['en', 'af', 'nl', 'xh', 'zu']
SOURCE_PATH = 'data'
TARGET_PATH = 'data/normalised'

def run_normalisation():
    """
Normalises all training files for each language by applying text preprocessing and saving the output to a target folder.
"""
    for lang in LANGUAGE_CODES:
        input_filename = f'train.{lang}.txt'   # fixed extension
        input_path = os.path.join(SOURCE_PATH, input_filename)
        output_path = os.path.join(TARGET_PATH, input_filename)
        print(f'Normalising {input_path} → {output_path}')
        normalise_text_file(input_path, output_path)

LANGS = ['en', 'af', 'nl', 'xh', 'zu']
TRAIN_DIR = 'data/normalised'
VAL_DIR = 'data'
GENERATED_OUTPUT_LEN = 200
SMOOTHING_ALPHA = 0.3

def train_all_models():
    """
Trains a character-level trigram language model for each language using normalised training data.

Returns:
    dict: A dictionary mapping language codes to their respective trained TrigramLM models.
"""
    models = {}
    for lang in LANGS:
        model = TrigramLM(delta=SMOOTHING_ALPHA)
        train_path = os.path.join(TRAIN_DIR, f'train.{lang}.txt')
        print(f' Training model for {lang} from {train_path}...')
        model.ingest_file(train_path)
        models[lang] = model
    return models

def generate_samples(models):
    """
Generates and prints a sample text sequence from each trained language model.
"""
    print("\n Sample generations:")
    for lang, model in models.items():
        print(f'\n--- {lang.upper()} ---')
        print(model.synth(limit=GENERATED_OUTPUT_LEN))

def compute_perplexity_matrix(models):
    """
Computes and prints a 5×5 cross-lingual perplexity matrix using the trained models and validation sets.
Also saves the matrix as a CSV file and plots a heatmap as a PNG.
"""
    print("\n Perplexity matrix:")
    results = {} 
    for model_lang, model in models.items():
        results[model_lang] = {}
        for val_lang in LANGS:
            val_file = os.path.join(VAL_DIR, f'val.{val_lang}.txt')
            perp = model.perplexity(val_file)
            results[model_lang][val_lang] = round(perp, 2)
    header = 'Model\\Validation'.ljust(15) + ''.join(f'{l.upper():>10}' for l in LANGS)
    print(header)
    for m_lang in LANGS:
        row = m_lang.upper().ljust(15)
        for v_lang in LANGS:
            row += f'{results[m_lang][v_lang]:>10}'
        print(row)
    df = pd.DataFrame(results).T
    df.to_csv('perplexity_matrix.csv', index=True)
    plotter.figure(figsize=(8, 6))
    sea.heatmap(df, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plotter.title("Cross-Language Perplexity Matrix")
    plotter.xlabel("Validation Language")
    plotter.ylabel("Model Language")
    plotter.tight_layout()
    plotter.savefig('perplexity_heatmap.png')

def run_all_models(Models, sentence):
    """
    Runs all language models on a sentence and returns the language with lowest perplexity and all scores.
    """
    min_perplex = float('inf')
    best_lang = None
    all_perplexities = {}
    for lang, model in models.items(): 
        perplexity = model.perplexity_from_string(sentence)
        all_perplexities[lang] = perplexity 
        if perplexity < min_perplex:
            min_perplex = perplexity
            best_lang = lang
    return best_lang, all_perplexities

def display_th_trigrams(model, context='th'):
    """
    Display all trigram continuations for the given 2-character history, e.g., 'th'.
    Shows P(c | context) for each possible continuation character c.
    """
    if not model.ready:
        raise RuntimeError("Model is not trained.")
    print(f"\nTrigram continuations for context '{context}':")
    history = context
    prob_dist = {}
    for c in model.symbols:  
        p = model.prob(history, c)
        if p > 0:
            prob_dist[c] = p
    if not prob_dist:
        print("No valid trigrams found with this history.")
        return
    for char, p in sorted(prob_dist.items(), key=lambda x: -x[1]):
        print(f"P({char} | {context}) = {p:.6f}")

def language_identification(models):
    """
    Performs language identification on test set using perplexity, outputs accuracy and saves predictions to CSV.
    """
    Prediction = namedtuple('Prediction', ['true_label', 'predicted_label', 'true_perplexity', 'predicted_perplexity', 'sentence'])
    test_res = []
    with open('data/test.lid.txt', 'r', encoding='utf-8') as file:
        for line in file:
            true_label = line[:2]
            sentence = line[3:].strip()
            if sentence == '':
                continue
            predicted_lang, all_perplexities = run_all_models(models, sentence)
            true_perplexity = all_perplexities[true_label]
            predicted_perplexity = all_perplexities[predicted_lang]
            test_res.append(Prediction(true_label, predicted_lang, true_perplexity, predicted_perplexity, sentence))
    print("\nLanguage Identification Results written to a file:")
    correct = sum(p.true_label == p.predicted_label for p in test_res)
    accuracy = correct / len(test_res)
    print(f"\nOverall Accuracy: {accuracy:.2%}")
    with open('language_identification_results.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['true_label', 'predicted_label', 'true_perplexity', 'predicted_perplexity', 'sentence'])
        for pred in test_res:
            writer.writerow([
                pred.true_label,
                pred.predicted_label,
                f"{pred.true_perplexity:.4f}",
                f"{pred.predicted_perplexity:.4f}",
                pred.sentence
            ])

if __name__ == '__main__':
    run_normalisation()
    models = train_all_models()
    generate_samples(models)
    compute_perplexity_matrix(models)
    display_th_trigrams(models['en'], context='th')
    language_identification(models)
    print("\n Running post-analysis script")
    subprocess.run(['python', 'analyse_identification_results.py'])
    print("\n Running BPE analysis")
    run_bpe_analysis()
