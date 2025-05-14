import os
import csv
import sys
from instrument_recognition import make_instrument_pred, init_inst_recog

def analysis(path):
    instrum_label, instrum_score = make_instrument_pred(path)
    print(f"Instrument label: {instrum_label}, Score: {instrum_score}")
    return instrum_label.lower()

def find_audio_files(directory):
    """
    Recursively find .wav and .mp3 files in the directory,
    skipping:
      - files that contain 'mix' in the name (case-insensitive)
      - hidden files (like .DS_Store)
      - files whose parent folder is named 'other' (case-insensitive)
    """
    audio_files = []
    for root, _, files in os.walk(directory):
        parent_folder = os.path.basename(root)
        if parent_folder.lower() == 'other':
            continue  # Skip entire folder

        for file in files:
            if file.startswith('.'):
                continue  # Skip hidden/system files

            if file.lower().endswith(('.mp3', '.wav')):
                full_path = os.path.join(root, file)
                audio_files.append((parent_folder, full_path))
    return audio_files

def write_results(audio_files, output_file='results_moises.csv'):
    """Write the audio file info and analysis result to a CSV file."""
    with open(output_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['label', 'path', 'prediction'])
        for parent_folder, path in audio_files:
            score = analysis(path)
            writer.writerow([parent_folder, path, score])

def main():
    init_inst_recog()
    if len(sys.argv) != 2:
        print("Usage: python analyze_moisesdb.py <directory_path>")
        sys.exit(1)

    directory = sys.argv[1]

    if not os.path.isdir(directory):
        print(f"Error: '{directory}' is not a valid directory.")
        sys.exit(1)

    audio_files = find_audio_files(directory)
    write_results(audio_files)
    print(f"Analysis complete. Results written to 'results_moises.csv'.")

if __name__ == "__main__":
    main()
