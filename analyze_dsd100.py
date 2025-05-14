import os
import csv
import sys
from instrument_recognition import make_instrument_pred, init_inst_recog

def analysis(path):    
    instrum_label, instrum_score = make_instrument_pred(path)
    print(f"Instrument label: {instrum_label}, Score: {instrum_score}")
    return instrum_label.lower()

def find_audio_files(directory):
    """Recursively find .wav and .mp3 files in the directory, skipping files that contain 'mix' (case-insensitive)."""
    audio_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            name, ext = os.path.splitext(file)
            if 'mix' in name.lower() or 'other' in name.lower():
                continue
            if ext.lower().endswith(('.mp3', '.wav')):
                full_path = os.path.join(root, file)
                audio_files.append((name, full_path))  # store name without extension
    return audio_files

def write_results(audio_files, output_file='results_dsd100.csv'):
    """Write the audio file info and analysis result to a CSV file."""
    with open(output_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['label', 'path', 'prediction'])
        for filename, path in audio_files:
            score = analysis(path)
            writer.writerow([filename, path, score])

def main():
    init_inst_recog()
    if len(sys.argv) != 2:
        print("Usage: python analyze_audio.py <directory_path>")
        sys.exit(1)

    directory = sys.argv[1]

    if not os.path.isdir(directory):
        print(f"Error: '{directory}' is not a valid directory.")
        sys.exit(1)

    audio_files = find_audio_files(directory)
    write_results(audio_files)
    print(f"Analysis complete. Results written to 'results_dsd100.csv'.")

if __name__ == "__main__":
    main()
