# BERGAMOT

This repository provides models, source code, and data for **BERGAMOT**: Biomedical Entity Representation with Graph-Augmented Multi-Objective Transformer which will be presented at [NAACL 2024](https://2024.naacl.org/).

![BERGAMOT](fig/bergamot.jpg)

# Overview

**TODO**

# Data

**TODO**

# Pre-trained model

GAT-BERGAMOT: [https://huggingface.co/andorei/BERGAMOT-multilingual-GAT](https://huggingface.co/andorei/BERGAMOT-multilingual-GAT):

```
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("andorei/BERGAMOT-multilingual-GAT")
model = AutoModel.from_pretrained("andorei/BERGAMOT-multilingual-GAT")
```


# Citation

**TODO**
