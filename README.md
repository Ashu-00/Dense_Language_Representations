# Dense_Language_Representations

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



