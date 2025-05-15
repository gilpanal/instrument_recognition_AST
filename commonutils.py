import csv
from instrument_recognition import make_instrument_pred, getaudioexcerpt
from annotation_utils import THRESHOLD_SPEECH_PRED_SCORE
from audio_analysis import tellifisspeech

def analysis(path, lookup):
    print(f"Analyzing file: {path}")
    is_speech_pred = tellifisspeech(path)
    print(f"Is speech prediction: {is_speech_pred}")
    if is_speech_pred < THRESHOLD_SPEECH_PRED_SCORE:
        processedpath =  getaudioexcerpt(path)
        instrum_label, instrum_score = make_instrument_pred(processedpath, is_speech_pred)
        print(f"Instrument label: {instrum_label}, Score: {instrum_score}")
        proposed = map_to_proposed_label(instrum_label, lookup)
        return instrum_label.lower(), proposed
    else:
        return "speech", "speech"

# Flatten the LABEL_CATEGORIES dict into a reverse lookup map
def build_label_lookup(label_categories):
    """Build a lookup dictionary from label to category."""
    return {
        label.lower(): category
        for category, aliases in label_categories.items()
        for label in aliases
    }

def map_to_proposed_label(label, label_lookup):
    """Map an input label to its high-level category using a lookup map."""
    return label_lookup.get(label.lower(), "other")


# OPTION 1: write to file after all results are obtained
def write_results(audio_files, lookup, output_file='results_moises.csv'):
    """Write the audio file info and analysis result to a CSV file."""
    with open(output_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['label', 'path', 'prediction', 'proposed_label'])
        for filename, path in audio_files:
            prediction, proposed = analysis(path, lookup)
            writer.writerow([filename, path, prediction, proposed])

# OPTION 2: write to file as the results are obtained
def write_results_streaming(audio_files, lookup, output_file='results_moises.csv'):
    with open(output_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['label', 'path', 'prediction', 'proposed_label'])
        f.flush()  # flush header immediately
        for filename, path in audio_files:
            try:
                prediction, proposed = analysis(path, lookup)
            except Exception as e:
                print(f"Error processing {path}: {e}")
                prediction, proposed = "error", "error"
            writer.writerow([filename, path, prediction, proposed])
            f.flush()  # flush after each row