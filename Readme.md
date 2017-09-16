# A simple sentence encoding-based method for Text Entailment on SICK dataset

## Dependencies
To run it perfectly, you will need:
* Python 2.7
* Theano 0.8.2

## Running the Script
1. Preprocess to get dictionary from training set
```
cd data
python build_dictionary.py
```

2. Train and test model
```
cd scripts/sent_enc_nli/
bash train.sh
```

The result is in `log.txt` file.