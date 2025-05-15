import csv
import sys
import os

def get_unique_labels(csv_path):
    if not os.path.isfile(csv_path):
        print(f"Error: File '{csv_path}' does not exist.")
        sys.exit(1)

    unique_labels = set()

    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row['label'].strip().lower()
            unique_labels.add(label)

    return sorted(unique_labels)

def main():
    if len(sys.argv) != 2:
        print("Usage: python list_labels.py /path/to/results.csv")
        sys.exit(1)

    csv_path = sys.argv[1]
    labels = get_unique_labels(csv_path)
    print("Unique labels found:")
    print(labels)

if __name__ == "__main__":
    main()
