from datasets import load_dataset
import gensim.downloader as api
import numpy as np
import matplotlib.pyplot as plt
import os


models = [
        "glove-wiki-gigaword-50",
        "glove-wiki-gigaword-100",
        "glove-wiki-gigaword-300",
        "word2vec-google-news-300",
        "fasttext-wiki-news-subwords-300"
    ]

ds = load_dataset("fairnlp/weat", "words")


def get_embedding(word_list, word2id, word_embeddings):
    return np.array([word_embeddings[word2id[word]] if word in word2id else None for word in word_list])

if __name__ == "__main__":
    if not os.path.exists("static_bias_results.json"):
        ds = ds["words"]
        print(ds)
        for id, words in zip(ds["id"], ds["words"]):
            if id == "female_attributes":
                fem_attr = words
                continue
            if id == "male_attributes":
                male_attr = words
                continue
            
            if id == "career_words":
                career_words = words
                continue
            if id == "family_words":
                family_words = words
                continue
        results = {}
        for model in models:
            print(f"Evaluating for model: {model}")
            word_vectors = api.load(model)
            
            word2id = {word: idx for idx, word in enumerate(word_vectors.index_to_key)}
            id2word = {idx: word for word, idx in word2id.items()}
            word_embeddings = word_vectors.vectors

            fem_attr_emb = get_embedding(fem_attr, word2id, word_embeddings)
            male_attr_emb = get_embedding(male_attr, word2id, word_embeddings)
            career_words_emb = get_embedding(career_words, word2id, word_embeddings)
            family_words_emb = get_embedding(family_words, word2id, word_embeddings)



            #similarity between female and career
            sim_fem_car = np.dot(fem_attr_emb, career_words_emb.T) / (np.linalg.norm(fem_attr_emb) * np.linalg.norm(career_words_emb))

            #similarity between male and career
            sim_male_car = np.dot(male_attr_emb, career_words_emb.T) / (np.linalg.norm(male_attr_emb) * np.linalg.norm(career_words_emb))

            #similarity between female and family
            sim_fem_fam = np.dot(fem_attr_emb, family_words_emb.T) / (np.linalg.norm(fem_attr_emb) * np.linalg.norm(family_words_emb))

            #similarity between male and family
            sim_male_fam = np.dot(male_attr_emb, family_words_emb.T) / (np.linalg.norm(male_attr_emb) * np.linalg.norm(family_words_emb))

            print("Model:", model)
            print("Female-Career Similarity:", sim_fem_car.mean())
            print("Male-Career Similarity:", sim_male_car.mean())
            print("Female-Family Similarity:", sim_fem_fam.mean())
            print("Male-Family Similarity:", sim_male_fam.mean())
            print()

            results[model] = {
                "sim_fem_car": float(sim_fem_car.mean()),
                "sim_male_car": float(sim_male_car.mean()),
                "sim_fem_fam": float(sim_fem_fam.mean()),
                "sim_male_fam": float(sim_male_fam.mean())
            }

            #save heatmaps
            os.makedirs("heatmaps", exist_ok=True)
            plt.figure(figsize=(10, 8))
            plt.subplot(2, 2, 1)
            plt.title("Female-Career Similarity")
            plt.imshow(sim_fem_car, cmap='hot', interpolation='nearest')
            plt.colorbar()
            plt.xticks(ticks=np.arange(len(career_words)), labels=career_words, rotation=90, fontsize=8)
            plt.yticks(ticks=np.arange(len(fem_attr)), labels=fem_attr, fontsize=8)

            plt.subplot(2, 2, 2)
            plt.title("Male-Career Similarity")
            plt.imshow(sim_male_car, cmap='hot', interpolation='nearest')
            plt.colorbar()
            plt.xticks(ticks=np.arange(len(career_words)), labels=career_words, rotation=90, fontsize=8)
            plt.yticks(ticks=np.arange(len(male_attr)), labels=male_attr, fontsize=8)

            plt.subplot(2, 2, 3)
            plt.title("Female-Family Similarity")
            plt.imshow(sim_fem_fam, cmap='hot', interpolation='nearest')
            plt.colorbar()
            plt.xticks(ticks=np.arange(len(family_words)), labels=family_words, rotation=90, fontsize=8)
            plt.yticks(ticks=np.arange(len(fem_attr)), labels=fem_attr, fontsize=8)

            plt.subplot(2, 2, 4)
            plt.title("Male-Family Similarity")
            plt.imshow(sim_male_fam, cmap='hot', interpolation='nearest')
            plt.colorbar()
            plt.xticks(ticks=np.arange(len(family_words)), labels=family_words, rotation=90, fontsize=8)
            plt.yticks(ticks=np.arange(len(male_attr)), labels=male_attr, fontsize=8)
            plt.tight_layout()
            plt.savefig(f"heatmaps/{model}_similarity.png")
            plt.close()
        
        #save results to json
        import json
        with open("static_bias_results.json", "w") as f:
            json.dump(results, f, indent=4)
    else:
        import json
        with open("static_bias_results.json", "r") as f:
            results = json.load(f)
    
    #plot graphs

    plt.figure(figsize=(10, 12))
    # vs career
    plt.subplot(2, 1, 1)
    models_names = list(results.keys())
    fem_car_sims = [results[model]["sim_fem_car"] for model in models_names]
    male_car_sims = [results[model]["sim_male_car"] for model in models_names]

    x = np.arange(len(models_names))
    width = 0.2
    plt.bar(x- width/2, fem_car_sims, width=width)
    plt.bar(x + width/2, male_car_sims, width=width)
    plt.xticks(x, models_names, rotation=45, ha='right', fontsize=8)
    plt.title("Similarity with Career Words")
    plt.ylabel("Cosine Similarity")
    plt.legend(["Female Attributes", "Male Attributes"])
    plt.grid()

    # vs family
    plt.subplot(2, 1, 2)
    fem_fam_sims = [results[model]["sim_fem_fam"] for model in models_names]
    male_fam_sims = [results[model]["sim_male_fam"] for model in models_names]

    x = np.arange(len(models_names))
    width = 0.2
    plt.bar(x- width/2, fem_fam_sims, width=width)
    plt.bar(x + width/2, male_fam_sims, width=width)
    plt.xticks(x, models_names, rotation=45, ha='right', fontsize=8)
    plt.title("Similarity with Family Words")
    plt.ylabel("Cosine Similarity")
    plt.legend(["Female Attributes", "Male Attributes"])
    plt.grid()
    plt.tight_layout()
    plt.savefig("bias_comparison.png")
    plt.close()