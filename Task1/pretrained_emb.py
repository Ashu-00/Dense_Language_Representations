from representation import analogy_task, simlex, read_file_lines
import gensim.downloader as api
from tqdm import tqdm
import numpy as np
import json
import os

if __name__ == "__main__":
    models = [
        "glove-wiki-gigaword-50",
        "glove-wiki-gigaword-100",
        "glove-wiki-gigaword-300",
        "word2vec-google-news-300",
        "fasttext-wiki-news-subwords-300"
    ]

    corr = {}
    acc = {}

    for model in models:
        print(f"Evaluating for model: {model}")
        word_vectors = api.load(model)
        word2id = {word: idx for idx, word in enumerate(word_vectors.index_to_key)}
        id2word = {idx: word for word, idx in word2id.items()}
        word_embeddings = word_vectors.vectors

        # Analogy task evaluation
        acc[model] = analogy_task(word2id, id2word, word_embeddings)

        # SimLex-999 evaluation
        corr[model] = simlex(word2id, word_embeddings)

        print("-----")
        print(f"Model: {model}")
        print(f"Analogy Task Accuracy: {acc[model]:.2f}")
        print(f"SimLex-999 Spearman Correlation: {corr[model]:.2f}")

    #save to json
        if os.path.exists("evaluation_results.json"):
            with open("evaluation_results.json", "r", encoding='utf-8') as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    existing_data = {}
        else:
            existing_data = {}
            existing_data["Analogy Task"] = {}
            existing_data["SimLex-999"] = {}

        # Update the existing data structure
        existing_data["Analogy Task"].update({model: acc[model]})
        existing_data["SimLex-999"].update({model: corr[model]})

        # Write the combined data back to the file
        with open("evaluation_results.json", "w", encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=4)
    
