import json
import numpy as np
import essentia
import essentia.streaming as ess
import essentia.standard as es
from essentia.standard import MonoLoader, TensorflowPredictVGGish, TensorflowPredictMusiCNN, TensorflowPredict2D
from instrument_recognition import init_inst_recog
from annotation_utils import SILENCE_RMS_DB_THRESHOLD


import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

sr_16 = 16000
sr_44 = 44100

# Hide logging messages from Essentia
essentia.log.infoActive = False

model_genre_rosamerica = TensorflowPredictVGGish(graphFilename='models/genre_rosamerica-vggish-audioset-1.pb')

def computeRMS(fullpath):
    audio_loader = MonoLoader()
    audio_loader.configure(filename=fullpath)
    audio = audio_loader()
    rms = es.RMS()(audio)
    rms_db = 20 * np.log10(rms + 1e-10) # Add small epsilon to avoid log(0)
    return rms_db

def tellifsilence(fullpath):
    rms_db = computeRMS(fullpath)
    # Check if the RMS value is below the threshold
    #if rms < 0.01:
    # Other possible value : -40 dB
    if rms_db < SILENCE_RMS_DB_THRESHOLD:
        return True, rms_db
    else:
        return False, rms_db

def tellifisspeech(fullpath):
    audio_sr16_loader = MonoLoader()
    audio_sr16_loader.configure(filename=fullpath, sampleRate=sr_16, resampleQuality=4)
    audio_sr16 = audio_sr16_loader()
    predictions = model_genre_rosamerica(audio_sr16)
    predictions = np.mean(predictions, axis=0)
    if isinstance(predictions,np.ndarray):
        return predictions[7]*100
    else:
        return 0