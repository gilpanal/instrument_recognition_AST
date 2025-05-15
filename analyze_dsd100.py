import os
import sys
from instrument_recognition import init_inst_recog
from commonutils import  write_results_streaming, build_label_lookup
from computeaccuracy import compute_accuracy

# Define a single dictionary mapping each label variant to its category
LABEL_CATEGORIES_DSD = {
    "vocals": [
        "singing", "mantra", "male singing", "female singing", 
        "child singing", "synthetic singing", "choir", "yodeling", 
        "chant", "humming", "rapping", "a capella"
    ],
    "drums": [
        "drum kit", "percussion", "drum machine", "drum", "snare drum", 
        "rimshot", "drum roll", "bass drum", "timpani", "tabla", 
        "cymbal", "hi-hat", "tambourine", "wood block"
    ],
    "bass": [
        "bass guitar", "double bass"
    ]
}

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

def main():
    
    if len(sys.argv) != 2:
        print("Usage: python analyze_audio.py <directory_path>")
        sys.exit(1)

    directory = sys.argv[1]
    output_file = 'results_dsd100.csv'

    if not os.path.isdir(directory):
        print(f"Error: '{directory}' is not a valid directory.")
        sys.exit(1)

    init_inst_recog()
    global lookup
    lookup = build_label_lookup(LABEL_CATEGORIES_DSD)
    audio_files = find_audio_files(directory)
    # OPTION 2: Uncomment the line below to write results as they are obtained    
    write_results_streaming(audio_files, lookup, output_file=output_file)
    # OPTION 1: Uncomment the line below to write results after all analysis
    # write_results(audio_files, lookup, output_file=output_file)
    print(f"Analysis complete. Results written to '{output_file}'.")
    
    compute_accuracy(output_file)  # <-- Call the accuracy function

if __name__ == "__main__":
    main()
