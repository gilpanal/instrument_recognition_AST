import os
import csv
import sys
from instrument_recognition import make_instrument_pred, init_inst_recog, getaudioexcerpt
from annotation_utils import THRESHOLD_SPEECH_PRED_SCORE
from audio_analysis import tellifsilence, tellifisspeech

# TODO: create groups for 
# ['bass', 'bowed_strings', 'drums', 'guitar', 'other_keys', 'other_plucked', 'percussion', 'piano', 'vocals', 'wind']
# Define a single dictionary mapping each label variant to its category
LABEL_CATEGORIES = {
    "vocals": [
        "singing", "mantra", "male singing", "female singing", 
        "child singing", "synthetic singing", "choir", "yodeling", 
        "chant", "humming", "rapping", "a capella"
    ],
    "drums": [
        "drum kit", "percussion", "drum machine", "drum", "snare drum", 
        "rimshot", "drum roll", "bass drum", "timpani", "tabla", 
        "cymbal", "hi-hat", "tambourine"
    ],
    "bass": [
        "bass guitar", "double bass"
    ],
    "piano": [
        "piano", "electric piano", "keyboard (musical)"
    ],
    "guitar": [
        "acoustic guitar", "electric guitar", "guitar", "steel guitar, slide guitar", "tapping (guitar technique)"
    ],
    "wind": [
        "flute", "saxophone", "trumpet", "trombone", "clarinet", "wind instrument, woodwind instrument", "brass instrument", "french horn"
    ],
    "bowed_strings": [
        "violin, fiddle", "cello", "double bass", "bowed string instrument"
    ],
    "other_keys": [
        "synthesizer", "organ", "harpsichord", "electronic organ", "hammond organ", "sampler"
    ],
    "other_plucked": [
        "harp", "banjo", "mandolin", "sitar", "zither", "ukulele", "plucked string instrument"
    ],
    "percussion": [
        "marimba, xylophone", "vibraphone", "steelpan", "wood block"
    ]
}

# Flatten the LABEL_CATEGORIES dict into a reverse lookup map
LABEL_LOOKUP = {
    label.lower(): category
    for category, aliases in LABEL_CATEGORIES.items()
    for label in aliases
}

def map_to_proposed_label(label):
    return LABEL_LOOKUP.get(label.lower(), "other")

def analysis(path):
    print(f"Analyzing file: {path}")
    is_speech_pred = tellifisspeech(path)
    print(f"Is speech prediction: {is_speech_pred}")
    if is_speech_pred < THRESHOLD_SPEECH_PRED_SCORE:
        processedpath =  getaudioexcerpt(path)
        instrum_label, instrum_score = make_instrument_pred(processedpath, is_speech_pred)
        print(f"Instrument label: {instrum_label}, Score: {instrum_score}")
        proposed = map_to_proposed_label(instrum_label)
        return instrum_label.lower(), proposed
    else:
        return "speech", "speech"

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

# OPTION 1: write to file after all results are obtained
def write_results(audio_files, output_file='results_moises.csv'):
    """Write the audio file info and analysis result to a CSV file."""
    with open(output_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['label', 'path', 'prediction', 'proposed_label'])
        for filename, path in audio_files:
            prediction, proposed = analysis(path)
            writer.writerow([filename, path, prediction, proposed])

# OPTION 2: write to file as the results are obtained
def write_results_streaming(audio_files, output_file='results_moises.csv'):
    with open(output_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['label', 'path', 'prediction', 'proposed_label'])
        f.flush()  # flush header immediately
        for filename, path in audio_files:
            try:
                prediction, proposed = analysis(path)
            except Exception as e:
                print(f"Error processing {path}: {e}")
                prediction, proposed = "error", "error"
            writer.writerow([filename, path, prediction, proposed])
            f.flush()  # flush after each row

def compute_accuracy(results_file='results_moises.csv'):
    total = 0
    correct = 0
    with open(results_file, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            if row['label'].strip().lower() == row['proposed_label'].strip().lower():
                correct += 1
    if total == 0:
        print("No data to evaluate.")
    else:
        accuracy = (correct / total) * 100
        print(f"Accuracy: {accuracy:.2f}% ({correct}/{total} matches)")

def main():
    
    if len(sys.argv) != 2:
        print("Usage: python analyze_moisesdb.py <directory_path>")
        sys.exit(1)

    directory = sys.argv[1]
    output_file = 'results_moises.csv'

    if not os.path.isdir(directory):
        print(f"Error: '{directory}' is not a valid directory.")
        sys.exit(1)
    
    init_inst_recog()
    audio_files = find_audio_files(directory)
    # OPTION 2: Uncomment the line below to write results as they are obtained    
    write_results_streaming(audio_files, output_file=output_file)
    # OPTION 1: Uncomment the line below to write results after all analysis
    # write_results(audio_files, output_file=output_file)
    print(f"Analysis complete. Results written to '{output_file}'.")

    compute_accuracy(output_file) # <-- Call the accuracy function

if __name__ == "__main__":
    main()
