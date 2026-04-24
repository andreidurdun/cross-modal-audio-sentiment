import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse

LABEL_NAMES = ["unsatisfied", "neutral", "satisfied"]

def plot_confusion_matrix(cm, title, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
                annot_kws={"size": 16})
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, format='pdf')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot confusion matrix from test_results.json")
    parser.add_argument('--input', type=str, required=True, help='Path to test_results.json')
    parser.add_argument('--output', type=str, required=True, help='Path to output PDF')
    parser.add_argument('--title', type=str, default='Confusion Matrix', help='Title for the plot')
    args = parser.parse_args()

    with open(args.input, 'r') as f:
        data = json.load(f)
    
    # Handle different JSON structures
    if 'confusion_matrix' in data:
        cm = np.array(data['confusion_matrix'])
    elif 'metrics' in data and 'confusion_matrix' in data['metrics']:
        cm = np.array(data['metrics']['confusion_matrix'])
    else:
        raise KeyError("Could not find confusion_matrix in the JSON file")
    
    plot_confusion_matrix(cm, args.title, args.output)
    print(f"Saved plot to {args.output}")

if __name__ == "__main__":
    main()