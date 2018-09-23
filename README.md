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
    unzip -j data/glove.6B.zip glove.6B.100d.txt -d embeddings
    ```
2. Get CoNLL-2005 data in the right format using [this repo](https://github.com/strubell/preprocess-conll05). 
Follow the instructions all the way through [preprocessing for evaluation](https://github.com/strubell/preprocess-conll05#pre-processing-for-evaluation-scripts).
3. **Make sure `data_dir` is set correctly, to the root directory of the data, in any config files you wish to use.**

Train a model:
----
To train a model with save directory `model` using the configuration file `lisa-conll05.conf`:
```bash
python network.py --config_file config/lisa-conll05.conf --save_dir model
```

Evaluate a model:
----
To evaluate the model saved in the directory `model`:
```bash
python network.py --load --test --config_file model/config.cfg --load_dir model
```