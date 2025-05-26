```bash
git clone https://github.com/gilpanal/instrument_recognition_AST.git

cd instrum_recogn_results

python3 -m venv venv

. venv/bin/activate

pip install -r requirements.txt


python analyze_dsd100.py /path/to/dsd100

python analyze_moisesdb.py /path/to/moisesdb


python list_labels.py results_dsd100subset.csv

python computeaccuracy.py results_dsd100subset.csv

```

