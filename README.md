# LISA: Linguistically-Informed Self-Attention

This is the original implementation of the linguistically-informed self-attention (LISA) model 
described in the following paper:
> Emma Strubell, Patrick Verga, Daniel Andor, David Weiss, and Andrew McCallum. [Linguistically-Informed 
> Self-Attention for Semantic Role Labeling](https://arxiv.org/abs/1804.08199). 
> *Conference on Empirical Methods in Natural Language Processing (EMNLP)*. 
> Brussels, Belgium. October 2018. 

This code is based on a fork of Timothy Dozat's 
[open source graph-based dependency parser](https://github.com/tdozat/Parser-v1). Thanks Tim!

This code is released only for exact replication of the paper. 
**You can find a vastly improved re-implementation of LISA [here](https://github.com/strubell/LISA).**

Requirements:
----
- Python 2.7
- \>= TensorFlow 1.1

Quick start:
============

Data setup (CoNLL-2005):
----
1. Get pre-trained word embeddings (GloVe):
    ```bash
    wget -P data http://nlp.stanford.edu/data/glove.6B.zip
    unzip -j data/glove.6B.zip glove.6B.100d.txt -d data/glove
    ```
2. Get CoNLL-2005 data in the right format using [this repo](https://github.com/strubell/preprocess-conll05). 
Follow the instructions all the way through [preprocessing for evaluation](https://github.com/strubell/preprocess-conll05#pre-processing-for-evaluation-scripts).
3. **Make sure `data_dir` is set correctly, to the root directory of the data, in any config files you wish to use below.**

Download and evaluate a pre-trained model:
----
Pre-trained models are available for download [here](https://drive.google.com/drive/u/1/folders/1E0Jn05VFqZTbbVcDoM5DEIHEFzD91iLs).

To download the model from Google Drive on the command line, you can use the following bash function (from [here](https://gist.github.com/iamtekeste/3cdfd0366ebfd2c0d805)):
```bash
function gdrive_download () {
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
  rm -rf /tmp/cookies.txt
}
```
The `gdrive_download` function takes two parameters: Google Drive id and filename. You can find model names and their
corresponding ids in this table:

| Model         | ID                                |
| ------------- | --------------------------------- |
| sa-conll05    | 1Qj5idT0A-OQlN24HRoHAt9G2P-CrW9zb |
| lisa-conll05  | 1XW3qWWMONnQt0lPyj0_t5KdcD7hF8k4F |
| sa-conll12    | 1kGAs73-5HtM9UY4IAHYmFXiyYDQomzo5 |
| lisa-conll12  | 1V9-lB-TrOxvProqiJ5Tx1cFUdHkTyjbw |

To download and evaluate e.g. the `lisa-conll05` model:
```bash
mkdir -p models
gdrive_download 1XW3qWWMONnQt0lPyj0_t5KdcD7hF8k4F models/lisa-conll05.tar.gz
tar xzvf models/lisa-conll05.tar.gz -C models
python network.py --load --test --test_eval --load_dir models/lisa-conll05 --config_file models/lisa-conll05/config.cfg 
```

Run the D&M+ELMo parser
----
TODO

Train a model:
----
We highly recommend using a GPU. 

To train a model with save directory `model` using the configuration file `lisa-conll05.conf`:
```bash
python network.py --config_file config/lisa-conll05.conf --save_dir model
```

