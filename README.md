## About instrument_recognition_AST

This repository evaluates the accuracy of the [AST (Audio Spectrogram Transformer) model](https://github.com/YuanGongND/ast) for instrument recognition on multitrack music datasets. The main entry points are `analyze_dsd100.py` and `analyze_moisesdb.py`, which run AST-based labeling on local dataset copies and compare the predicted labels against dataset ground truth.

The code in the [`AST`](AST) directory is adapted from the instrument-recognition pipeline used in the [Hi-Audio platform](https://hiaudio.fr). It combines:

- AST for instrument label prediction
- [Essentia](https://essentia.upf.edu/essentia_python_tutorial.html) for MIR-related preprocessing and speech/singing detection
- dataset-specific label mappings so AST predictions can be compared with DSD100 and MoisesDB labels

The analysis scripts scan the selected dataset, infer a predicted AST label for each source track, map that label to a dataset-compatible category, and write the result to a CSV file.

For DSD100, only the labels `bass`, `vocals`, and `drums` are evaluated. The AST labels are grouped as follows:

```python
LABEL_CATEGORIES_DSD = {
    "vocals": [
        "singing", "mantra", "male singing", "female singing",
        "child singing", "synthetic singing", "choir", "yodeling",
        "chant", "humming", "rapping", "a capella", "vocal music"
    ],
    "drums": [
        "drum kit", "percussion", "drum machine", "drum", "snare drum", 
        "rimshot", "drum roll", "bass drum", "timpani", "tabla", 
        "cymbal", "hi-hat", "tambourine", "wood block"
    ],
    "bass": [
        "bass guitar", "double bass", "synthesizer", "sampler"
    ]
}
```

MoisesDB uses a broader mapping with categories such as `bass`, `bowed_strings`, `drums`, `guitar`, `other_keys`, `other_plucked`, `percussion`, `piano`, `vocals`, and `wind`. See the [MoisesDB dictionary](analyze_moisesdb.py) for the full mapping.

The repository includes example result files: `results_dsd100.csv` and `results_moises.csv`. Generated CSV files contain four columns:

1. `label`: the original label proposed in the dataset (ground truth)
2. `path`: the location of the file that is analyzed
3. `prediction`: the default output label from the AST model
4. `proposed_label`: the dataset-equivalent label derived from the AST `prediction`, using the mapping defined for that dataset (e.g. `"singing"` maps to `"vocals"` for DSD100)

The following table presents the accuracy metrics obtained through AST-based analysis of multitrack music datasets (DSD100 and MoisesDB) for one-class classification.

| Dataset    | AST accuracy                 |
|------------|------------------------------|
| `DSD100`   | 92.00% (276/300 matches)     |
| `MoisesDB` | 87.29% (2219/2542 matches)   |


## Download the datasets

- **DSD100:** https://sigsep.github.io/datasets/dsd100.html
- **MoisesDB:** https://music.ai/research/


## Prerequisites

- Python 3.10
- A local copy of DSD100 and/or MoisesDB (see above)
- `ffmpeg` available on your system if your dataset contains `.mp3` files, as `pydub` uses it for audio decoding

This repository already includes the model files used by the scripts:

- `AST/pretrained_models/audio_mdl.pth`
- `AST/models/genre_rosamerica-vggish-audioset-1.pb`

No additional model download step is required.

> **Note:** `essentia-tensorflow` can be the trickiest dependency to install depending on your platform. If it fails, follow the platform-specific instructions in the [Essentia documentation](https://essentia.upf.edu/installing.html).


## Run the code locally

```bash
git clone https://github.com/gilpanal/instrument_recognition_AST.git

cd instrument_recognition_AST

python3 -m venv venv

. venv/bin/activate

pip install -r requirements.txt
```

### Analyze the datasets

```bash
python analyze_dsd100.py /path/to/dsd100

python analyze_moisesdb.py /path/to/moisesdb
```

Each script:

- recursively scans the dataset directory for `.wav` and `.mp3` files
- writes results to `results_dsd100.csv` or `results_moises.csv`
- prints the computed accuracy summary at the end of the run

Dataset-specific assumptions:

- `analyze_dsd100.py` skips files whose name contains `mix` or `other`; the ground-truth label is inferred from the file name
- `analyze_moisesdb.py` skips hidden files and folders named `other`; the ground-truth label is inferred from the parent folder name

### Example of output for DSD100 AST analysis

![screenshot](doc/output_analysis.png)


### Get the list of unique labels in results file

```bash
python AST/list_labels.py results_dsd100.csv

python AST/list_labels.py results_moises.csv
```

### Example of output for MoisesDB listing unique labels script

![screenshot](doc/output_list_labels.png)

### Compute accuracy of the results obtained for one dataset

```bash
python computeaccuracy.py results_dsd100.csv

python computeaccuracy.py results_moises.csv
```

`computeaccuracy.py` reports the overall accuracy and lists all mismatched files.

### Example of output for DSD100 AST accuracy computation including mismatches

![screenshot](doc/output_accuracy.png)


## More info about Hi-Audio

1. Journal article: https://hal.science/hal-05153739v1
2. Hi-Audio online platform: https://hiaudio.fr
3. News: https://hi-audio.imt.fr/2025/03/07/bridging-music-and-research/
4. Hi-Audio back-end repository: https://github.com/idsinge/hiaudio_backend


---

## Acknowledgements

This work is developed as part of the project *Hybrid and Interpretable Deep Neural Audio Machines*, funded by the **European Research Council (ERC)** under the European Union's Horizon Europe research and innovation programme (grant agreement No. 101052978).

<img src="./doc/ERC_logo.png" alt="European Research Council logo" width="250"/>

We also thank [Teysir Baoueb](https://github.com/Teysir-B) (@Teysir-B) for proposing the AST model as a basis for musical instrument recognition and for her contributions during the early-stage proof of concept that laid the groundwork for this repository.


---

## How to Cite

If you use or reference the data or findings from this repository, please cite the published journal article. You may also cite the repository directly.

> Gil Panal, J. M., David, A., & Richard, G. (2026). The Hi-Audio online platform for recording and distributing multi-track music datasets. *Journal on Audio, Speech, and Music Processing*. https://doi.org/10.1186/s13636-026-00459-0

**BibTeX:**

```bibtex
@article{GilPanal2026,
  author  = {Gil Panal, Jos{\'e} M. and David, Aur{\'e}lien and Richard, Ga{\"e}l},
  title   = {The Hi-Audio online platform for recording and distributing multi-track music datasets},
  journal = {Journal on Audio, Speech, and Music Processing},
  year    = {2026},
  issn    = {3091-4523},
  doi     = {10.1186/s13636-026-00459-0},
  url     = {https://doi.org/10.1186/s13636-026-00459-0}
}
```

A preprint version is also available at: [https://hal.science/hal-05153739](https://hal.science/hal-05153739)

**Repository citation:**

> Gil Panal, J. M., David, A., & Richard, G. (2026). *Instrument Recognition with AST* [Software repository]. GitHub. https://github.com/gilpanal/instrument_recognition_AST

```bibtex
@misc{GilPanal2026ast,
  author = {Gil Panal, Jos{\'e} M. and David, Aur{\'e}lien and Richard, Ga{\"e}l},
  title  = {Instrument Recognition with AST},
  year   = {2026},
  url    = {https://github.com/gilpanal/instrument_recognition_AST}
}
```


---

## License

This project is licensed under the [MIT License](LICENSE).  
Copyright (c) 2026 Hi-Audio.
