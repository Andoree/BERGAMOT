# BERGAMOT: Biomedical Entity Representation with Graph-Augmented Multi-Objective Transformer

This repository provides models, source code, and data for **BERGAMOT**: **B**iomedical **E**ntity **R**epresentation with **G**raph-**A**ugmented **M**ulti-**O**bjective **T**ransformer which will be presented at [NAACL 2024](https://2024.naacl.org/).

![BERGAMOT](fig/bergamot.jpg)

The model supports all the languages available in UMLS version 2020AB: English, Spanish, French, Dutch, German, Finnish, Russian, Turkish, Korean, Chinese, Japanese, Thai, Portuguese, Italian, Swedish, Hungarian, Polish, Estonian, Croatian, Ukrainian, Greek, Danish, Hebrew.

Here is the poster of our [paper presented at NAACL 2024](https://aclanthology.org/2024.findings-naacl.288.pdf):

<p align="center">
<img src="https://github.com/Andoree/BERGAMOT/blob/main/BERGAMOT_poster_naacl.jpg" width="800">
</p>



# Evaluation

To run zero-shot evaluation as described in our NAACL paper, you need to download the evaluation [data](https://github.com/AIRI-Institute/medical_crossing). To  run the evaluation, use the [eval_bert_ranking](https://github.com/alexeyev/Fair-Evaluation-BERT/tree/de40551e21f4bc2d38eb40d658f14a705cd596d7) script:
```python
python Fair-Evaluation-BERT/eval_bert_ranking.py --model_dir "andorei/BERGAMOT-multilingual-GAT" \
    --data_folder "data_medical_crossing/datasets/mantra/es/DISO-fair_exact_vocab" \
    --vocab "data_medical_crossing/vocabs/mantra_es_dict_DISO.txt"
```

# Environment

Required libraries are listed in [requirements.txt](https://github.com/Andoree/BERGAMOT/blob/main/requirements.txt). We ran our experiments using Python 3.8.

# Pre-trained model

GAT-BERGAMOT: [https://huggingface.co/andorei/BERGAMOT-multilingual-GAT](https://huggingface.co/andorei/BERGAMOT-multilingual-GAT):

```
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("andorei/BERGAMOT-multilingual-GAT")
model = AutoModel.from_pretrained("andorei/BERGAMOT-multilingual-GAT")
```


# Citation
```bibtex
@inproceedings{sakhovskiy-et-al-2024-bergamot,
    title = "Biomedical Entity Representation with Graph-Augmented Multi-Objective Transformer",
    author = "Sakhovskiy, Andrey and Semenova, Natalia and Kadurin, Artur and Tutubalina, Elena",
    booktitle = "Findings of the Association for Computational Linguistics: NAACL 2024",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
  }
```
