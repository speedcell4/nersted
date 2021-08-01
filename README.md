# NERsted

Implementation
of [Nested Named Entity Recognition via Explicitly Excluding the Influence of the Best Path](https://aclanthology.org/2021.acl-long.275/).

## Requirements

- Python 3.8 or higher
- `python3 -m pip install -r requirements.txt`

## Preprocessing

After obtaining all these files, please move them to the `data` directory, and it is supposed to look like the following,

```
data
├── ace2004
│   ├── ace2004.dev
│   ├── ace2004.test
│   └── ace2004.train
├── ace2005
│   ├── ace2005.dev
│   ├── ace2005.test
│   └── ace2005.train
├── genia
│   ├── genia.dev
│   ├── genia.test
│   └── genia.train
│── biobert_large
│   ├── config.json
│   ├── model.ckpt.data-00000-of-00001
│   ├── model.ckpt.index
│   ├── model.ckpt.meta
│   └── vocab.txt
└── pubmed
    └── PubMed-shuffle-win-2.bin
```

### Data Preprocessing

We strictly followed the data preprocessing procedure
of [Shibuya and Hovy (2020)](https://aclanthology.org/2020.tacl-1.39/), please
follow [their script](https://github.com/yahshibu/nested-ner-tacl2020-transformers).

### Embedding Preprocessing

For ACE2004 and ACE2005 datasets, our model utilizes `torchglyph` library to automatically download, cache and load
Glove vectors.

For the GENIA dataset, please download and extract `PubMed-shuffle-win-2.bin`
from [BioNLP-2016](https://github.com/cambridgeltl/BioNLP-2016#word-vectors).

### BERT Preprocessing

For ACE2004 and ACE2005 datasets, our model utilizes `transformers` library to automatically download the corresponding
checkpoints.

For the GENIA dataset, please download `BioBERT-Large v1.1 (+ PubMed 1M)` and extract checkpoint and config files
from [BioBERT Pre-trained Weights](https://github.com/naver/biobert-pretrained#downloading-pre-trained-weights).

### Flair Preprocessing

For ACE2004, ACE2005, and GENIA datasets, our model utilizes `flair` library to automatically download the corresponding
checkpoints.

## Training

The first run will download, extract, and cache files, i.e., Glove and BERT vectors, so it may take a long time.

### ACE2004 dataset

```shell
CUDA_VISIBLE_DEVICES=0 python3 -m ner --data ace2004 \
  --study ace2004-naive \
  --dec NaiveDecoder \
  --lr 0.03 --gamma 0.02 --momentum 0.9 \
  --use_bert false --use_flair false
  
CUDA_VISIBLE_DEVICES=0 python3 -m ner --data ace2004 \
  --study ace2004-max \
  --dec MaxDecoder \
  --lr 0.03 --gamma 0.02 --momentum 0.9 \
  --use_bert false --use_flair false
  
CUDA_VISIBLE_DEVICES=0 python3 -m ner --data ace2004 \
  --study ace2004-lse \
  --dec LoSumExpDecoder \
  --lr 0.03 --gamma 0.02 --momentum 0.9 \
  --use_bert false --use_flair false
```

Turn on `--use_bert` and/or `--use_flair` when you want to employ BERT and/or Flair.

### ACE2005 dataset

```shell
CUDA_VISIBLE_DEVICES=0 python3 -m ner --data ace2005 \
  --study ace2005-naive \
  --dec NaiveDecoder \
  --lr 0.04 --gamma 0.02 --momentum 0.9 \
  --use_bert false --use_flair false
  
CUDA_VISIBLE_DEVICES=0 python3 -m ner --data ace2005 \
  --study ace2005-max \
  --dec MaxDecoder \
  --lr 0.04 --gamma 0.02 --momentum 0.9 \
  --use_bert false --use_flair false
  
CUDA_VISIBLE_DEVICES=0 python3 -m ner --data ace2005 \
  --study ace2005-lse \
  --dec LoSumExpDecoder \
  --lr 0.04 --gamma 0.02 --momentum 0.9 \
  --use_bert false --use_flair false
```

Turn on `--use_bert` and/or `--use_flair` when you want to employ BERT and/or Flair.

### GENIA dataset

```shell
CUDA_VISIBLE_DEVICES=0 python3 -m ner --data genia \
  --study genia-naive \
  --dec NaiveDecoder \
  --lr 0.02 --gamma 0.03 --momentum 0.9 \
  --use_bert false --use_flair false
  
CUDA_VISIBLE_DEVICES=0 python3 -m ner --data genia \
  --study genia-max \
  --dec MaxDecoder \
  --lr 0.02 --gamma 0.03 --momentum 0.9 \
  --use_bert false --use_flair false
  
CUDA_VISIBLE_DEVICES=0 python3 -m ner --data genia \
  --study genia-lse \
  --dec LoSumExpDecoder \
  --lr 0.02 --gamma 0.03 --momentum 0.9 \
  --use_bert false --use_flair false
```

Turn on `--use_bert` and/or `--use_flair` when you want to employ BERT and/or Flair.

## Citation

```
@inproceedings{wang-etal-2021-nested,
    title = "Nested Named Entity Recognition via Explicitly Excluding the Influence of the Best Path",
    author = "Wang, Yiran  and
      Shindo, Hiroyuki  and
      Matsumoto, Yuji  and
      Watanabe, Taro",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.275",
    doi = "10.18653/v1/2021.acl-long.275",
    pages = "3547--3557",
    abstract = "This paper presents a novel method for nested named entity recognition. As a layered method, our method extends the prior second-best path recognition method by explicitly excluding the influence of the best path. Our method maintains a set of hidden states at each time step and selectively leverages them to build a different potential function for recognition at each level. In addition, we demonstrate that recognizing innermost entities first results in better performance than the conventional outermost entities first scheme. We provide extensive experimental results on ACE2004, ACE2005, and GENIA datasets to show the effectiveness and efficiency of our proposed method.",
}
```