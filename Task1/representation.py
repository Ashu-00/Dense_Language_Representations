import os
from tqdm import tqdm
import numpy as np
import json
import scipy.sparse #sparse matrix due to OOM
import sklearn.decomposition


WINDOW_SIZE = 40
DIMENSION_FALLBACK = 100

def read_file_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()

def remove_starting_numbers(lines):
    cleaned_lines = []
    for line in lines:
        cleaned_line = line.lstrip('0123456789. \t').rstrip().lower()
        cleaned_lines.append(cleaned_line)
    return cleaned_lines

def remove_special_characters(lines, remove_numbers=False):
    cleaned_lines = []
    for line in lines:
        if remove_numbers:
            line = ''.join(char for char in line if not char.isdigit())
        cleaned_line = ''.join(char for char in line if char.isalnum() or char.isspace())
        cleaned_lines.append(cleaned_line)
    return cleaned_lines

def clean_corpus(file_path, remove_numbers=False):
    lines = read_file_lines(file_path)
    cleaned_lines = remove_starting_numbers(lines)
    cleaned_lines = remove_special_characters(cleaned_lines, remove_numbers=remove_numbers)
    return cleaned_lines

def create_vocabulary(lines):
    vocabulary = set()
    for line in lines:
        words = line.split()
        vocabulary.update(words)
    return vocabulary

def create_tokens(lines, word2id):
    tokenized_lines = []
    for line in tqdm(lines):
        words = line.split()
        tokenized_line = []
        for word in (words):
            tokenized_line.append(word2id.get(word, -1)) # -1 for unk
        tokenized_lines.append(tokenized_line)
    return tokenized_lines

def create_coocc_matrix(tok_lines, vocab_size, window_size=WINDOW_SIZE):
    # print(vocab_size)
    coocc_matrix = scipy.sparse.lil_matrix((vocab_size, vocab_size), dtype=np.int32)
    # print(coocc_matrix.shape)
    for tok_line in tqdm(tok_lines):
        line_length = len(tok_line)
        for i, target_id in enumerate(tok_line):
            if target_id == -1:
                continue
            start = max(0, i - window_size)
            end = min(line_length, i + window_size + 1)
            for j in range(start, end):
                if i != j:
                    context_id = tok_line[j]
                    if context_id != -1:
                        coocc_matrix[target_id, context_id] += 1
    return coocc_matrix.tocsr() 

def dimen_redn(coocc_matrix, target_dim=DIMENSION_FALLBACK):

    svd = sklearn.decomposition.TruncatedSVD(
        n_components=target_dim, 
        algorithm='randomized', 
        random_state=42,
        n_iter=7
    )

    uk_sk = svd.fit_transform(coocc_matrix)
    print(type(uk_sk), uk_sk.shape)
    Sk = svd.singular_values_
    reduced_matrix = uk_sk / np.sqrt(Sk + 1e-9)
    return reduced_matrix

def PCA_reduce(embeddings, target_dim=2):
    pca = sklearn.decomposition.PCA(n_components=target_dim, random_state=42)
    reduced_embeddings = pca.fit_transform(embeddings)
    return reduced_embeddings

def tsne_reduce(embeddings, target_dim=2, num_words=10):
    tsne = sklearn.manifold.TSNE(n_components=target_dim, random_state=42, init='random', perplexity=min(30, num_words-1))
    reduced_embeddings = tsne.fit_transform(embeddings)
    return reduced_embeddings

def plot_emb_pca(sample_words, word2id,):
    sample_indices = [word2id.get(word, -1) for word in sample_words]
    sample_embeddings = word_embeddings[sample_indices]
    reduced_embeddings = PCA_reduce(sample_embeddings, target_dim=2)
    # reduced_embeddings = tsne_reduce(sample_embeddings, target_dim=2, num_words=len(sample_words))
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    for i, word in enumerate(sample_words):
        x, y = reduced_embeddings[i]
        plt.scatter(x, y)
        plt.text(x + 0.01, y + 0.01, word, fontsize=5)
    plt.title("PCA of Sample Word Embeddings")
    plt.xlabel("PCA Dimension 1")
    plt.ylabel("PCA Dimension 2")
    plt.grid()
    plt.savefig("sample_word_embeddings_pca.png")

def simlex(word2id, word_embeddings):
    if os.path.exists("SimLex-999/SimLex-999/SimLex-999.txt"):
        print("Evaluating on SimLex-999")
        from scipy.stats import spearmanr
        simlex_lines = read_file_lines("SimLex-999/SimLex-999/SimLex-999.txt")[1:] #skip header
        human_scores = []
        model_scores = []
        for line in simlex_lines:
            parts = line.strip().split('\t')
            word1 = parts[0]
            word2 = parts[1]
            score = float(parts[3])
            if word1 in word2id and word2 in word2id:
                vec1 = word_embeddings[word2id[word1]]
                vec2 = word_embeddings[word2id[word2]]
                cosine_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                human_scores.append(score)
                model_scores.append(cosine_sim)
        correlation, _ = spearmanr(human_scores, model_scores)
        print(f"Spearman correlation for SimLex-999: {correlation:.4f}")
        return correlation
    return None

def analogy_task(word2id, id2word, word_embeddings):
    if os.path.exists("archive/questions-words.txt"):
        print("Evaluating on analogy task")

        norm_embeddings = word_embeddings / (np.linalg.norm(word_embeddings, axis=1, keepdims=True) + 1e-9)

        analogy_lines = read_file_lines("archive/questions-words.txt")
        correct = 0
        total = 0
        for line in tqdm(analogy_lines):
            if line.startswith(":"):
                continue
            line = line.lower()
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            a, b, c, d = parts
            if a in word2id and b in word2id and c in word2id and d in word2id:

                id_a = word2id[a]
                id_b = word2id[b]
                id_c = word2id[c]
                id_d = word2id[d]


                vec_a = word_embeddings[id_a]
                vec_b = word_embeddings[id_b]
                vec_c = word_embeddings[id_c]



                predicted_vec = vec_b - vec_a + vec_c
                pred_vec_norm = predicted_vec / (np.linalg.norm(predicted_vec) + 1e-9)
                similarities = np.dot(norm_embeddings, pred_vec_norm)
                similarities[id_a] = -1.0 
                similarities[id_b] = -1.0
                similarities[id_c] = -1.0
                predicted_id = np.argmax(similarities)
                predicted_word = id2word[predicted_id]
                if predicted_word == d:
                    correct += 1
                total += 1
        accuracy = correct / total if total > 0 else 0
        print(f"Analogy task accuracy: {accuracy:.4f}")
        return accuracy
    return None


if __name__ == "__main__":
    if not os.path.exists("eng_word2id.json"):
        print("Reading corpus")
        corpus = read_file_lines("eng_corpora.txt")
        print("Total lines: ",len(corpus))
        print("Sample line: ",corpus[0])
        del corpus

        print("Cleaning corpus")
        cleaned_lines = clean_corpus("eng_corpora.txt", remove_numbers=True)
        print("Example line: ",cleaned_lines[0])

        print("Creating vocabulary")
        vocabulary = create_vocabulary(cleaned_lines)
        vocab_size = len(vocabulary)
        print("Vocabulary size: ", vocab_size)

        print("Tokenizing") 
        word2id = {word: idx for idx, word in enumerate(vocabulary)}
        id2word = {idx: word for word, idx in word2id.items()}
        

        #save as json
        with open("eng_word2id.json", "w", encoding='utf-8') as f:
            json.dump(word2id, f, ensure_ascii=False, indent=4)
        with open("eng_id2word.json", "w", encoding='utf-8') as f:
            json.dump(id2word, f, ensure_ascii=False, indent=4)
    
        tokenised = create_tokens(cleaned_lines, word2id)
        print("Tokenized line: ", tokenised[0])

        #save as json
        with open("eng_tokenized.json", "w", encoding='utf-8') as f:
            json.dump(tokenised, f, ensure_ascii=False, indent=4)
    
    else:
            print("Loading existing resources")
            with open("eng_word2id.json", "r", encoding='utf-8') as f:
                word2id = json.load(f)
                word2id = {k: int(v) for k, v in word2id.items()}
            with open("eng_id2word.json", "r", encoding='utf-8') as f:
                id2word = json.load(f)
                id2word = {int(k): v for k, v in id2word.items()}
            with open("eng_tokenized.json", "r", encoding='utf-8') as f:
                tokenised = json.load(f)
            vocab_size = len(word2id)
    
    if not os.path.exists("eng_cooccurrence_matrix.npz"):
        print("Creating co-occurrence matrix")
        coocc_matrix = create_coocc_matrix(tokenised, vocab_size, window_size=WINDOW_SIZE)
        print("Co-occurrence matrix shape: ", coocc_matrix.shape)
        scipy.sparse.save_npz("eng_cooccurrence_matrix.npz", coocc_matrix)
    else:
        print("Loading existing co-occurrence matrix")
        coocc_matrix = scipy.sparse.load_npz("eng_cooccurrence_matrix.npz")
        print("Co-occurrence matrix shape: ", coocc_matrix.shape)
    print("Some values from co-occurrence matrix: ", coocc_matrix[60833, :5].toarray())
    print("total non-zero entries ", coocc_matrix.nnz)

    corr = {}
    acc = {}
    for dim in tqdm([32, 64, 256, 512, 1024]):
        print("-------")
        print("Dimension", dim)
        print("Performing dimensionality reduction")
        word_embeddings = dimen_redn(coocc_matrix, target_dim=dim)
        print("Word embeddings shape: ", word_embeddings.shape)
        
        #visualize some embeddings in plot by PCA
        print("Plotting sample word embeddings using PCA")
        sample_words = ["king", "queen", "man", "woman", "apple", "banana", "city", "village", "car", "bus"]
        plot_emb_pca(sample_words, word2id)

        #simlex 999 evaluation
        corr[dim] = simlex(word2id, word_embeddings)

        #analogy task evaluation
        acc[dim] = analogy_task(word2id, id2word, word_embeddings)
    
    print("Final Results:")
    print("Dimensions vs SimLex-999 Correlation:")
    for dim in corr:
        print(f"Dimension: {dim}, Spearman Correlation: {corr[dim]:.4f}")
    print("Dimensions vs Analogy Task Accuracy:")
    for dim in acc:
        print(f"Dimension: {dim}, Analogy Accuracy: {acc[dim]:.4f}")
    
    #save to json without overwriting
    with open("evaluation_results.json", "r+", encoding='utf-8') as f:
        try:
            existing_data = json.load(f)
        except json.JSONDecodeError:
            existing_data = {}

        if "SimLex-999" not in existing_data:
            existing_data["SimLex-999"] = {}
        for dim, correlation in corr.items():
            existing_data["SimLex-999"][f"emb_{dim}_{WINDOW_SIZE}"] = correlation

        if "Analogy Task" not in existing_data:
            existing_data["Analogy Task"] = {}
        for dim, accuracy in acc.items():
            existing_data["Analogy Task"][f"emb_{dim}_{WINDOW_SIZE}"] = accuracy

        # Write updated data back to the file
        f.seek(0)
        json.dump(existing_data, f, ensure_ascii=False, indent=4)
        f.truncate()
        
    


    


    




    

    

    

    
