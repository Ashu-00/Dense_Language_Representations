# Dense_Language_Representations

**⚠️ Note: Please read the accompanying report first for detailed methodology, analysis, and observations.**

## Project Description

### Task 1: Dense Representations

The objective of this module is to explore the foundations of distributional semantics by generating dense word embeddings from scratch. Instead of relying solely on pre-trained models, we construct embeddings using statistical methods:

* **Co-occurrence Matrix:** Converting a large English corpus (>300K sentences) into numerical form by tracking word pair frequencies within a fixed context window.
* **Dimensionality Reduction:** Applying techniques (such as SVD) to compress the sparse  matrix into efficient dense vectors ().
* **Evaluation:** Assessing embedding quality via semantic similarity, clustering, and visualization (t-SNE/PCA) against benchmarks like **SimLex-999** and **WordSimilarity-353**.
* **Benchmarking:** Comparing these statistical embeddings against neural methods like **Word2Vec**, **GloVe**, and **FastText**.

### Task 2: Cross-lingual Alignment

This module extends word representations to a multilingual setting, specifically aiming to align English and Hindi vector spaces. By mapping embeddings from two different languages into a shared space, we enable cross-lingual knowledge transfer.

* **Methodology:** We utilize **Procrustes Analysis** to learn a linear transformation matrix that aligns source (English) and target (Hindi) embeddings.
* **Evaluation:** The effectiveness of the alignment is quantitatively measured to ensure that semantically similar words across languages are mapped closely in the shared vector space.

### Bonus Task: Harmful Associations

Word embeddings trained on large-scale human corpora often inherit social biases (e.g., gender, race). This module focuses on the ethical evaluation of NLP models.

* **Static vs. Contextual:** We implement an evaluation regimen to quantify spurious associations in static embeddings (like GloVe) and analyze whether modern contextual models (like **BERT**) mitigate or perpetuate these biases.

---

**Important**: Install requirements.txt before everything. `pip install -r requirements.txt`

## Task 1

#### Codebase
- `representation.py`: Main code file consisting of all major functions for creating embeddings and evaluating them
- `pretrained_emb.py`: Evaluation of pretrained models
- `plot_results.py` : Plots the graphs from the evaluation results
- `check_dim_d.py`: To plot explained variance for getting best embedding dimension (is not required)

#### How-to-run

1. Run representation.py for embeddings creation and evaluation
2. Run pretrained_emb.py for evaluating other models
3. Plot the results using plot_results.py

#### Directory structure
```
Task1/
├── check_dim_d.py
├── eng_corpora.txt
├── evaluation_results.json
├── plot_results.py
├── pretrained_emb.py
├── representation.py
├── archive/
│   ├── questions-words.csv
│   └── questions-words.txt
└── SimLex-999/
    └── SimLex-999/
        ├── README.txt
        └── SimLex-999.txt
```

## Task 2

#### CodeBase

- `crossalign.py`: Main code for all training and evaluation
- `get_emb.py`: Run to download embeddings for english and hindi

#### How-to-run

1. Run the `get_emb.py`. (expects gunzip and curl)
2. Run `crossalign.py` to get all the results and save the transformation matrices.

#### Directory Structure

```
Task2/
├── alignment_results.json
├── crossalign.py
├── get_emb.sh
├── hi-en.0-5000.txt
└── hi-en.5000-6500.txt
```

## Bonus Task

#### Directory Structure
```
Bonus-Task/
├── contextual_emb/
│   ├── bias_results_th.1_.json
│   ├── bias_results.json
│   ├── contectual.py
│   └── plot_results.py
├── static_word_emb/
│   ├── static_bias_results.json
│   ├── static.py
│   └── heatmaps/
```

#### Codebase
- `static_word_emb/static.py`: Does the evaluation on static embeddings of given models and saves the results and heatmaps.

- `contextual_emb/contectual.py`: Evaluates and saves all the bias results.

- `contextual_emb/plot_results.py`: Plots the results of contextual models.



