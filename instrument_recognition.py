import csv
import sys
import torch, torchaudio
import numpy as np
from torch.amp import autocast
from essentia.standard import MonoLoader
import os
from pydub import AudioSegment
from pydub.silence import split_on_silence
from annotation_utils import LOWER_SPEECH_PRED_SCORE
from src.models.ast_models import ASTModel
from instrument_filtered_labels import instrument_labels
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import config

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
 # Load the AudioSet label set
label_csv = os.path.join(SCRIPT_DIR, 'egs/audioset/data', 'class_labels_indices.csv')
# label_csv = './egs/audioset/data/class_labels_indices.csv'       # label and indices for audioset data
 # Export result
export_dir = os.path.join(config.DATA_BASEDIR, "processed")
os.makedirs(export_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Assume each input spectrogram has 1024 time frames
input_tdim = 1024
labels = None
audio_model = None

# Create a new class that inherits the original ASTModel class
class ASTModelVis(ASTModel):
    def get_att_map(self, block, x):
        qkv = block.attn.qkv
        num_heads = block.attn.num_heads
        scale = block.attn.scale
        B, N, C = x.shape
        qkv = qkv(x).reshape(B, N, 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        return attn

    def forward_visualization(self, x):
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        B = x.shape[0]
        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        # save the attention map of each of 12 Transformer layer
        att_list = []
        for blk in self.v.blocks:
            cur_att = self.get_att_map(blk, x)
            att_list.append(cur_att)
            x = blk(x)
        return att_list

def make_features(wav_name, mel_bins, target_length=1024):

    sr_16 = 16000
    audio_sr16_loader = MonoLoader()
    audio_sr16_loader.configure(filename=wav_name, sampleRate=sr_16, resampleQuality=4)
    audio_sr16 = audio_sr16_loader()
    audio_tensor = torch.tensor(audio_sr16)
    audio_tensor = audio_tensor.to(device)
    audio_tensor = audio_tensor.unsqueeze(0)

    fbank = torchaudio.compliance.kaldi.fbank(
        audio_tensor, htk_compat=True, sample_frequency=sr_16, use_energy=False,
        window_type='hanning', num_mel_bins=mel_bins, dither=0.0, frame_shift=10)   
   
    n_frames = fbank.shape[0]

    p = target_length - n_frames
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]

    fbank = (fbank - (-4.2677393)) / (4.5689974 * 2)
    return fbank


def load_label(label_csv):
    with open(label_csv, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        lines = list(reader)
    labels = []
    ids = []  # Each label has a unique id such as "/m/068hy"
    for i1 in range(1, len(lines)):
        id = lines[i1][1]
        label = lines[i1][2]
        ids.append(id)
        labels.append(label)
    return labels

def load_ast_model(device):
    # Create an AST model and load the AudioSet pretrained weights
    
    # checkpoint_path = './pretrained_models/audio_mdl.pth'
    checkpoint_path = os.path.join(SCRIPT_DIR, 'pretrained_models', 'audio_mdl.pth')
    
    # now load the visualization model
    ast_mdl = ASTModelVis(label_dim=527, input_tdim=input_tdim, imagenet_pretrain=False, audioset_pretrain=False, verbose=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    audiomodel = torch.nn.DataParallel(ast_mdl, device_ids=[0])
    audiomodel.load_state_dict(checkpoint)
    audiomodel = audiomodel.to(device)
    audiomodel.eval()
    return audiomodel       

def getaudioexcerpt(filepath):
    audio = AudioSegment.from_file(filepath).set_channels(1)
    # Split audio into chunks, removing silence
    chunks = split_on_silence(audio,
        min_silence_len=500,       # consider as silence if >500ms
        silence_thresh=audio.dBFS - 10,  # silence threshold relative to dBFS
        keep_silence=200,           # keep 200ms of silence at each cut
        seek_step = 100
    ) 
    # Combine the chunks back into one audio file
    processed_audio = sum(chunks)
    duration_seconds = len(processed_audio) / 1000
    processedpath = os.path.join(export_dir, f"nosilence.wav")
    if duration_seconds > 60:
        # Extract 30 seconds after 10 seconds
        start_time = (30) * 1000  # convert to milliseconds
        end_time = (60) * 1000
        segment = processed_audio[start_time:end_time]
        segment.export(processedpath, format="wav")
    else:
        # Export full audio
        processed_audio.export(processedpath, format="wav")
    
    return processedpath

def make_instrument_pred(sample_audio_path, is_speech_pred):
    inst_label = None
    inst_score = 0
    feats = make_features(sample_audio_path, mel_bins=128)           # shape(1024, 128)
    feats_data = feats.expand(1, input_tdim, 128)           # reshape the feature
    feats_data = feats_data.to(device)
    # Alternative
    # feats_data = feats.unsqueeze(0).to(device)

    # do some masking of the input
    #feats_data[:, :512, :] = 0.

    # Make the prediction
    with torch.no_grad():
        if device.type == "cuda":
            with autocast("cuda"):
                output = audio_model.forward(feats_data)
                output = torch.sigmoid(output)
        else:
            output = torch.sigmoid(audio_model.forward(feats_data))
    result_output = output.data.cpu().numpy()[0]
    sorted_indexes = np.argsort(result_output)[::-1]

    # Print audio tagging top probabilities
    # print('Predice results:')
    # for k in range(10):
    #   print('- {}: {:.4f}'.format(np.array(labels)[sorted_indexes[k]], result_output[sorted_indexes[k]]))
       

    # Loop through sorted predictions to find the highest non-excluded label
    for idx in sorted_indexes:
        label = labels[idx]
        score = result_output[idx]
        # Skip "Speech" label if the is_speech_score is too low
        if label == "Speech" and is_speech_pred < LOWER_SPEECH_PRED_SCORE:
            continue
        if label in instrument_labels:
            inst_label = label
            inst_score = score
            break 

    return inst_label, inst_score

def init_inst_recog():
    global labels, audio_model
    labels = load_label(label_csv)
    audio_model = load_ast_model(device)
    print("Instrument recognition model initialized.")