# BERGAMOT: Biomedical Entity Representation with Graph-Augmented Multi-Objective Transformer

This repository provides models, source code, and data for **BERGAMOT**: **B**iomedical **E**ntity **R**epresentation with **G**raph-**A**ugmented **M**ulti-**O**bjective **T**ransformer which will be presented at [NAACL 2024](https://2024.naacl.org/).

![BERGAMOT](fig/bergamot.jpg)



# Evaluation

**TODO**

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
