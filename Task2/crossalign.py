import os
import re
from gensim.models import KeyedVectors
# from gensim.models import fasttext
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

def load_embeddings(embedding_path):
    model = KeyedVectors.load_word2vec_format(
        embedding_path, 
        binary=False, 
        limit=500000,
        unicode_errors='ignore',
    )
    return model

def load_dict(file):
    pairs = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            #since we want en-hi
            tgt_word, src_word = line.strip().split()

            pairs.append((src_word, tgt_word))
    return pairs

def normalize_embeddings(emb):
    emb -= np.mean(emb, axis=0)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return emb

def procustes(x,y):
    m = np.dot(x.T, y)
    u, s, vt = np.linalg.svd(m)
    return np.dot(u, vt)

#bilingual lexicon induction evaluation
def evaluate_alignment(en_model, hi_model, W):
    #load test dictionary
    test_dict_file = "hi-en.5000-6500.txt"
    test_word_pairs = load_dict(test_dict_file)

    precision_at_1 = 0
    precision_at_5 = 0
    total = 0

    for en_word, hi_word in test_word_pairs:
        if en_word in en_model and hi_word in hi_model:
            en_vec = en_model[en_word]
            en_vec = en_vec - np.mean(en_vec)
            en_vec = en_vec / np.linalg.norm(en_vec)

            en_vec_mapped = np.dot(en_vec, W)

            closeWords = hi_model.similar_by_vector(en_vec_mapped, topn=5)
            total += 1
            found = False
            for rank, (word, sim) in enumerate(closeWords):
                if word == hi_word:
                    found = True
                    if rank == 0:
                        precision_at_1 += 1
                    precision_at_5 += 1
                    break
            if not found:
                pass
    return precision_at_1 / total, precision_at_5 / total

def linear_reg(X,Y,epochs=100, lr=0.01):
    n, d = X.shape
    W = np.random.rand(d,d)

    #full batch gd
    loop = tqdm(range(epochs))
    for epoch in loop:
        Y_pred = np.dot(X, W)
        loss = np.mean((Y - Y_pred)**2)

        grad = -2 * (np.dot(X.T, (Y - Y_pred)) /n)

        W -= lr * grad

        if epoch % 100 == 0:
            loop.set_description(f"Epoch {epoch} Loss {loss:.4f}")

    return W

class disc(nn.Module):
    def __init__(self, input_dim):
        super(disc, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layer(x)

def adversial_training(X, Y, epochs=100, disc_lr=0.0001, gen_lr=0.1, ortho=False, ortho_lambda=0.1):
    n, d = X.shape

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    W = torch.randn(d, d, requires_grad=True)

    crit = disc(d)
    criterion = nn.BCELoss()
    optimD = optim.Adam(crit.parameters(), lr=disc_lr)
    optimW = optim.Adam([W], lr=gen_lr)
    real_labels = torch.ones(n, 1)
    fake_labels = torch.zeros(n, 1)
    identity = torch.eye(d)


    for epoch in range(epochs):
        optimD.zero_grad()
        optimW.zero_grad()

        X_mapped = torch.matmul(X, W)

        #train disc
        outputs_real = crit(Y)
        loss_real = criterion(outputs_real, real_labels)
        outputs_fake = crit(X_mapped.detach())
        loss_fake = criterion(outputs_fake, fake_labels)

        lossD = (loss_real + loss_fake) / 2
        lossD.backward()
        optimD.step()

        #train W
        optimW.zero_grad()
        X_mapped2 = torch.matmul(X, W)
        outputs_fake2 = crit(X_mapped2)

        lossG = criterion(outputs_fake2, real_labels)
        if ortho:
            WtW = torch.matmul(W.T, W)
            ortho_loss = torch.sum((WtW - identity)**2)
            lossG += ortho_lambda * ortho_loss
        lossG.backward()
        optimW.step()

        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{epochs}], Loss D: {lossD.item()}, Loss G: {lossG.item()}")


    return W.detach().numpy()

def run_extrinsic_eval(train_data, test_data, W):
    #step1: map eng embeddings
    #step2: train a classifier on (X_train, Y_train)
    #step3: evaluate on (X_test, Y_test) for eng emb transformed
    #step4: evaluate on (X_test, Y_test) for hi emb
    X_train_eng = np.dot(train_data["eng_emb"], W)
    Y_train = np.array(train_data["label"])

    X_test_eng = np.dot(test_data["eng_emb"], W)
    Y_test = np.array(test_data["label"])

    X_test_hi = np.array(test_data["hi_emb"])

    scaler = StandardScaler()
    X_train_eng_scaled = scaler.fit_transform(X_train_eng)
    X_test_eng_scaled = scaler.transform(X_test_eng)
    X_test_hi_scaled = scaler.transform(X_test_hi)

    cls = MLPClassifier(
        hidden_layer_sizes=(100, 50), 
        activation='relu', 
        solver='adam', 
        max_iter=100,
        random_state=42,
        batch_size=256,
        early_stopping=True,
        verbose=False
    )
    cls.fit(X_train_eng_scaled, Y_train)

    eng_acc = cls.score(X_test_eng_scaled, Y_test)
    hi_acc = cls.score(X_test_hi_scaled, Y_test)
    return eng_acc, hi_acc

def init_extrinsic_eval_data(eng_model, hi_model):
    from datasets import load_dataset

    dataset = load_dataset("mteb/IndicSentiment", "hi")
    total_data = dataset['test'].train_test_split(test_size=0.2, seed=42) #since train set is very less
    label_map = {"Negative":0, "Positive":1}

    train_data = total_data['train'].filter(lambda x: x['LABEL'] in label_map)
    test_data = total_data['test'].filter(lambda x: x['LABEL'] in label_map)

    def process_data(example):

        example['label'] = label_map[example['LABEL']]
        #lower and remove special chars
        example["ENGLISH REVIEW"] = example["ENGLISH REVIEW"].lower()
        example["ENGLISH REVIEW"] = re.sub(r'\W+', ' ', example["ENGLISH REVIEW"]) 
        example["INDIC REVIEW"] = example["INDIC REVIEW"].lower()
        example["INDIC REVIEW"] = re.sub(r'\W+', ' ', example["INDIC REVIEW"])

        #embed
        eng_tokens = example["ENGLISH REVIEW"].strip().split()
        hi_tokens = example["INDIC REVIEW"].strip().split()
        eng_embs = []
        hi_embs = []
        for token in eng_tokens:
            if token in eng_model:
                eng_embs.append(eng_model[token])
        for token in hi_tokens:
            if token in hi_model:
                hi_embs.append(hi_model[token])

        if eng_embs:
            avg_vec = np.mean(eng_embs, axis=0)
            avg_vec-=np.mean(avg_vec)
            norm = np.linalg.norm(avg_vec)
            if norm > 0:
                avg_vec /= norm
            example["eng_emb"] = avg_vec
        else:
            example["eng_emb"] = np.zeros(eng_model.vector_size)
        
        if hi_embs:
            avg_vec = np.mean(hi_embs, axis=0)
            avg_vec-=np.mean(avg_vec)
            norm = np.linalg.norm(avg_vec)
            if norm > 0:
                avg_vec /= norm
            example["hi_emb"] = avg_vec
        else:
            example["hi_emb"] = np.zeros(hi_model.vector_size)
        
        return example

    train_data = train_data.map(process_data)
    test_data = test_data.map(process_data)


    return train_data, test_data


if __name__ == "__main__":
        eng_emb = "cc.en.300.vec"
        hin_emb = "cc.hi.300.vec"
        dict_file = "hi-en.0-5000.txt"
        results = {}

        import time
        start_time = time.time()
        print("Loading English embedding")
        en_model = load_embeddings(eng_emb)
        en_model.vectors_norm = normalize_embeddings(en_model.vectors)
        print("Loading Hindi embedding")
        hi_model = load_embeddings(hin_emb)
        hi_model.vectors_norm = normalize_embeddings(hi_model.vectors)
        print("Loading dictionary")
        word_pairs = load_dict(dict_file)
        print("Total word pairs", len(word_pairs))
        print(f"Time taken to load embeddings and dictionary: {time.time() - start_time} seconds")

        emb_dim = en_model.vector_size
        print(f"Embedding dimension: {emb_dim}")


        # Keep only pairs where both words are in our loaded vocab
        X_train = []
        Y_train = []

        for en_word, hi_word in word_pairs:
                if en_word in en_model and hi_word in hi_model:
                    X_train.append(en_model[en_word])
                    Y_train.append(hi_model[hi_word])

        print(f"Number of word pairs found in both embeddings: {len(X_train)}")

        X_train = np.array(X_train)
        Y_train = np.array(Y_train)

        print("Normalizing embeddings")
        X_train = normalize_embeddings(X_train)
        Y_train = normalize_embeddings(Y_train)

        extrinsic_train , extrinsic_test = init_extrinsic_eval_data(en_model, hi_model)

        #testing on unaligned vectors for baseline
        W_unal = np.identity(emb_dim)
        print("Evaluating alignment on test dictionary for Unaligned embeddings")
        p_at_1, p_at_5 = evaluate_alignment(en_model, hi_model, W_unal)
        print(f"Precision@1: {p_at_1}")
        print(f"Precision@5: {p_at_5}")
        print("Running Extrinsic Evaluation (Sentiment classification) for Unaligned embeddings")
        eng_acc , hi_acc = run_extrinsic_eval(extrinsic_train, extrinsic_test, W_unal)
        print(f"English Test Accuracy (unaligned): {eng_acc}")
        print(f"Hindi Test Accuracy: {hi_acc}")
        print("_________\n")
        results['unaligned'] = {
            'precision@1': p_at_1,
            'precision@5': p_at_5,
            'eng_acc': eng_acc,
            'hi_acc': hi_acc
        }

        if not os.path.exists("alignment_matrix_procustes.npy"):

            print("Computing Procrustes alignment")
            W = procustes(X_train, Y_train)

            print("Saving alignment matrix. Shape:", W.shape)
            np.save("alignment_matrix_procustes.npy", W)
        else:
            print("Loading alignment matrix from file")
            W = np.load("alignment_matrix_procustes.npy")

        example_word = "thief"
        print("Example Test on", example_word)

        en_vec = en_model[example_word]
        en_vec = en_vec - np.mean(en_vec)
        en_vec = en_vec / np.linalg.norm(en_vec)

        print("Top 5 closest words in Hindi before mapping:")
        closeWords = hi_model.similar_by_vector(en_vec, topn=5)
        for word, sim in closeWords:
            print(f"{word}: {sim}")

        en_vec_mapped = np.dot(en_vec, W)

        closeWords = hi_model.similar_by_vector(en_vec_mapped, topn=5)
        print("Top 5 closest words in Hindi after mapping(procrustes):")
        for word, sim in closeWords:
            print(f"{word}: {sim}")

        print("Evaluating alignment on test dictionary for Procrustes alignment")
        p_at_1, p_at_5 = evaluate_alignment(en_model, hi_model, W)
        print(f"Precision@1: {p_at_1}")
        print(f"Precision@5: {p_at_5}")

        print("Running Extrinsic Evaluation (Sentiment classification) for Procrustes alignment")
        eng_acc, hi_acc = run_extrinsic_eval(extrinsic_train, extrinsic_test, W)
        print(f"English Test Accuracy (mapped): {eng_acc}")
        print(f"Hindi Test Accuracy: {hi_acc}")
        print("_________\n")

        results['procrustes'] = {
            'precision@1': p_at_1,
            'precision@5': p_at_5,
            'eng_acc': eng_acc,
            'hi_acc': hi_acc
        }

        if not os.path.exists("alignment_matrix_linear.npy"):
            print()
            print("Training Linear Regression alignment")

            W_linear = linear_reg(X_train, Y_train, epochs=3000, lr=0.1)

            print("Saving Linear Regression alignment matrix. Shape:", W_linear.shape)
            np.save("alignment_matrix_linear.npy", W_linear)
        else:
            print("Loading Linear Regression alignment matrix from file")
            W_linear = np.load("alignment_matrix_linear.npy")
        
        en_vec_mapped = np.dot(en_vec, W_linear)
        closeWords = hi_model.similar_by_vector(en_vec_mapped, topn=5)
        print("Top 5 closest words in Hindi after mapping(Linear Regression):")
        for word, sim in closeWords:
            print(f"{word}: {sim}")

        print("Evaluating alignment on test dictionary for Linear Regression alignment")
        p_at_1, p_at_5 = evaluate_alignment(en_model, hi_model, W_linear)
        print(f"Precision@1: {p_at_1}")
        print(f"Precision@5: {p_at_5}")

        print("Running Extrinsic Evaluation (Sentiment classification) for Linear Regression alignment")
        eng_acc, hi_acc = run_extrinsic_eval(extrinsic_train, extrinsic_test, W_linear)
        print(f"English Test Accuracy (mapped): {eng_acc}")
        print(f"Hindi Test Accuracy: {hi_acc}")
        print("_________\n")

        results['linear_regression'] = {
            'precision@1': p_at_1,
            'precision@5': p_at_5,
            'eng_acc': eng_acc,
            'hi_acc': hi_acc
        }

    
        if not os.path.exists("alignment_matrix_adversial.npy"):
            print("Training Adversial alignment")

            W_adv = adversial_training(X_train, Y_train, epochs=100, disc_lr=0.0001, gen_lr=0.01)

            print("Saving Adversial alignment matrix. Shape:", W_adv.shape)
            np.save("alignment_matrix_adversial.npy", W_adv)
        else:
            print("Loading Adversial alignment matrix from file")
            W_adv = np.load("alignment_matrix_adversial.npy")
        
        en_vec_mapped = np.dot(en_vec, W_adv)
        closeWords = hi_model.similar_by_vector(en_vec_mapped, topn=5)
        print("Top 5 closest words in Hindi after mapping(Adversial):")
        for word, sim in closeWords:
            print(f"{word}: {sim}")
        
        print("Evaluating alignment on test dictionary for Only Adversial alignment")
        p_at_1, p_at_5 = evaluate_alignment(en_model, hi_model, W_adv)
        print(f"Precision@1: {p_at_1}")
        print(f"Precision@5: {p_at_5}")

        print("Running Extrinsic Evaluation (Sentiment classification) for Adversial alignment")
        eng_acc, hi_acc = run_extrinsic_eval(extrinsic_train, extrinsic_test, W_adv)
        print(f"English Test Accuracy (mapped): {eng_acc}")
        print(f"Hindi Test Accuracy: {hi_acc}")
        print("_________\n")

        results['adversial_no_ortho'] = {
            'precision@1': p_at_1,
            'precision@5': p_at_5,
            'eng_acc': eng_acc,
            'hi_acc': hi_acc
        }

        print("Adversial + Procrustes Hybrid")
        W_ft = procustes(np.dot(X_train, W_adv), Y_train)
        W_hybrid = np.dot(W_adv, W_ft)
        en_vec_mapped = np.dot(en_vec, W_hybrid)

        closeWords = hi_model.similar_by_vector(en_vec_mapped, topn=5)
        print("Top 5 closest words in Hindi after mapping(Adversial + Procrustes):")
        for word, sim in closeWords:
            print(f"{word}: {sim}")
        print("Evaluating alignment on test dictionary for Adversial + Procrustes alignment")
        p_at_1, p_at_5 = evaluate_alignment(en_model, hi_model, W_hybrid)
        print(f"Precision@1: {p_at_1}")
        print(f"Precision@5: {p_at_5}")

        print("Running Extrinsic Evaluation (Sentiment classification) for Adversial + Procrustes alignment")
        eng_acc , hi_acc = run_extrinsic_eval(extrinsic_train, extrinsic_test, W_hybrid)
        print(f"English Test Accuracy (mapped): {eng_acc}")
        print(f"Hindi Test Accuracy: {hi_acc}")
        print("_________\n")

        results['hybrid_no_ortho'] = {
            'precision@1': p_at_1,
            'precision@5': p_at_5,
            'eng_acc': eng_acc,
            'hi_acc': hi_acc
        }

        print("Adversial + Procrustes Hybrid with orthogonality constraint")
        if not os.path.exists("alignment_matrix_adversial_ortho.npy"):
            print("Training Adversial alignment with orthogonality constraint")

            W_adv_ortho = adversial_training(X_train, Y_train, epochs=200, disc_lr=0.0001, gen_lr=0.1, ortho=True, ortho_lambda=1e-3)

            print("Saving Adversial alignment matrix with orthogonality constraint. Shape:", W_adv_ortho.shape)
            np.save("alignment_matrix_adversial_ortho.npy", W_adv_ortho)
        else:
            print("Loading Adversial alignment matrix with orthogonality constraint from file")
            W_adv_ortho = np.load("alignment_matrix_adversial_ortho.npy")

        W_ft_ortho = procustes(np.dot(X_train, W_adv_ortho), Y_train)
        W_hybrid_ortho = np.dot(W_adv_ortho, W_ft_ortho)
        en_vec_mapped = np.dot(en_vec, W_hybrid_ortho)
        closeWords = hi_model.similar_by_vector(en_vec_mapped, topn=5)
        print("Top 5 closest words in Hindi after mapping(Adversial with orthogonality + Procrustes ):")
        for word, sim in closeWords:
            print(f"{word}: {sim}")
        print("Evaluating alignment on test dictionary for Adversial with orthogonality + Procrustes alignment")
        p_at_1, p_at_5 = evaluate_alignment(en_model, hi_model, W_hybrid_ortho)
        print(f"Precision@1: {p_at_1}")
        print(f"Precision@5: {p_at_5}")
        print("Running Extrinsic Evaluation (Sentiment classification) for Adversial with orthogonality + Procrustes alignment")
        eng_acc , hi_acc = run_extrinsic_eval(extrinsic_train, extrinsic_test, W_hybrid_ortho)
        print(f"English Test Accuracy (mapped): {eng_acc}")
        print(f"Hindi Test Accuracy: {hi_acc}")
        print("_________\n")
        results['hybrid_with_ortho'] = {
            'precision@1': p_at_1,
            'precision@5': p_at_5,
            'eng_acc': eng_acc,
            'hi_acc': hi_acc
        }


        #save to json
        import json
        with open("alignment_results.json", "w") as f:
            json.dump(results, f, indent=4)
