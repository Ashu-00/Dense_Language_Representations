import numpy as np
import scipy.sparse
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import time


comp_to_test = 500

def run_truncated_svd(coocc_matrix, n_components=comp_to_test):
    svd = TruncatedSVD(n_components=n_components, n_iter=10, random_state=42)
    time_start = time.perf_counter()
    svd.fit(coocc_matrix)
    time_end = time.perf_counter()
    print(f"Truncated SVD fitting time: {time_end - time_start:.4f} seconds")
    return svd

if __name__ == "__main__":
    print("Loading co-occurrence matrix...")
    # Load the matrix you saved in representation.py
    coocc_matrix = scipy.sparse.load_npz("eng_cooccurrence_matrix.npz")
    print(f"Matrix loaded, shape: {coocc_matrix.shape}")

    svd = run_truncated_svd(coocc_matrix, n_components=comp_to_test)
    exp_var = svd.explained_variance_ratio_

    cum_var = np.cumsum(exp_var)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, comp_to_test + 1), cum_var, marker='.')
    plt.title('Cumulative Explained Variance by Number of Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid()
    plt.axhline(y=0.9, color='r', linestyle='--', label='90% Variance')
    plt.legend()
    plt.savefig("explained_variance.png")