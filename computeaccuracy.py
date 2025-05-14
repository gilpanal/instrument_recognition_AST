import csv
import sys
import os

def compute_accuracy(results_file):
    if not os.path.isfile(results_file):
        print(f"Error: File '{results_file}' does not exist.")
        sys.exit(1)

    total = 0
    correct = 0
    with open(results_file, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row['label'].strip().lower()
            proposed = row['proposed_label'].strip().lower()
            total += 1
            if label == proposed:
                correct += 1

    if total == 0:
        print("No data to evaluate.")
    else:
        accuracy = (correct / total) * 100
        print(f"Accuracy: {accuracy:.2f}% ({correct}/{total} matches)")

def main():
    if len(sys.argv) != 2:
        print("Usage: python computeaccuracy.py /path/to/results.csv")
        sys.exit(1)

    csv_path = sys.argv[1]
    compute_accuracy(csv_path)

if __name__ == "__main__":
    main()
