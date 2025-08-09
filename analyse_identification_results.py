import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plotter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#this script analyses the results of language identification using trigram language models.
# Computes perplexity for each language model on validation data and saves results to a CSV file.
# It also generates a confusion matrix and prints misclassifications ordered by confidence gap.
df = pd.read_csv("language_identification_results.csv")
correct = df['true_label'] == df['predicted_label']
accuracy = correct.mean()
print(f" Overall Accuracy: {accuracy:.2%}")
labels = sorted(df['true_label'].unique())
cm = confusion_matrix(df['true_label'], df['predicted_label'], labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap="Blues", values_format='d')
plotter.title("Language Identification Confusion Matrix")
plotter.tight_layout()
plotter.savefig("confusion_matrix.png")
mistakes = df[df['true_label'] != df['predicted_label']].copy()
mistakes['confidence_gap'] = (mistakes['predicted_perplexity'] - mistakes['true_perplexity']).abs()
mistakes_sorted = mistakes.sort_values(by='confidence_gap', ascending=True)
print("\n All Misclassifications (from least to most confident):")
print("These are ordered by the absolute perplexity gap between predicted and true language models.")
print("A small gap means the model was less confident—both models found the sentence similarly likely.")

for idx, row in mistakes_sorted.iterrows():
    print(f"\nTRUE: {row.true_label}, PRED: {row.predicted_label}")
    print(f"True Perp: {row.true_perplexity:.2f}, Pred Perp: {row.predicted_perplexity:.2f}, Gap: {row.confidence_gap:.2f}")
    print(f"Sentence: {row.sentence}")
