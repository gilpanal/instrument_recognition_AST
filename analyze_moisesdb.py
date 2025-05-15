import os
import sys
from AST.instrument_recognition import init_inst_recog
from commonutils import  write_results_streaming, build_label_lookup
from computeaccuracy import compute_accuracy

# TODO: create groups for 
# ['bass', 'bowed_strings', 'drums', 'guitar', 'other_keys', 'other_plucked', 'percussion', 'piano', 'vocals', 'wind']
# Define a single dictionary mapping each label variant to its category
LABEL_CATEGORIES_MOISES = {
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
    global lookup
    lookup = build_label_lookup(LABEL_CATEGORIES_MOISES)
    audio_files = find_audio_files(directory)
    # OPTION 2: Uncomment the line below to write results as they are obtained    
    write_results_streaming(audio_files, lookup, output_file=output_file)
    # OPTION 1: Uncomment the line below to write results after all analysis
    # write_results(audio_files, lookup, output_file=output_file)
    print(f"Analysis complete. Results written to '{output_file}'.")

    compute_accuracy(output_file) # <-- Call the accuracy function

if __name__ == "__main__":
    main()
