==================================================
NLP817 Assignment 1: Trigram Language Models & BPE
==================================================

This project trains character-level trigram language models for multiple languages, evaluates cross-lingual perplexity, implements Byte Pair Encoding (BPE) from scratch, and performs sentence-level language identification.

---

## Environment Setup

1. Ensure Python **3.13.2** is installed on your system.

2. Create and activate a virtual environment.

   ## • Using venv:

   python3.13 -m venv nlp817-env
   source nlp817-env/bin/activate # On macOS/Linux
   nlp817-env\Scripts\activate # On Windows

   ## • Using conda:

   conda create -n nlp817-env python=3.13.2
   conda activate nlp817-env

3. ## Install required dependencies:
   pip install -r requirements.txt

---

## Running the Project

Once your environment is ready, run the full pipeline with:

python main.py

This will:

- Preprocess and normalise the training corpora
- Train character-level trigram models on each language.
- Generate synthetic samples.
- Calculate and plot the perplexity matrix.
- Compute and output identification results and plots.
- Output predictions and accuracy to file.
- Run custom Byte Pair Encoding (BPE) on each language.
- Calculate vocabulary overlaps and generate BPE heatmap.

---

## Project Structure

├── main.py # Master script to run all steps (modelling, BPE, evaluation)
├── analyse_identification_results.py # Computes accuracy, confusion matrix for lang ID
├── bpeimplement.py # Custom Byte Pair Encoding and vocabulary overlap logic
├── check_val_files_for_nonbasic_chars.py # Script to verify validation/test character compliance
├── text_normaliser.py # Implements text cleaning pipeline for training data
├── triModel.py # Trigram model with smoothed probability estimation
├── tune_alpha.py # Script for tuning α (smoothing) based on validation perplexity
│

---

## Requirements

Python version: **3.13.2**

Install all required libraries from `requirements.txt`.

---

Once everything is set up, you're good to run:
python main.py

All results and visualisations will be generated automatically.
# Quantifying-Language-Identity-and-Similarity-Using-Trigrams-and-Subword-Overlap
