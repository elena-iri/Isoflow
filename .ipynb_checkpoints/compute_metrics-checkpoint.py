import pandas as pd
import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA
import ot   # POT: Python Optimal Transport
import argparse

# Function to compute MMD with RBF kernel
# (gaussian kernel that is the standard and most powerful- it's sensitive to sigma so we try different ones)
def rbf_kernel(X, Y, sigma):
    XX = np.sum(X**2, axis=1)[:, np.newaxis]
    YY = np.sum(Y**2, axis=1)[np.newaxis, :]
    distances = XX + YY - 2 * X @ Y.T
    return np.exp(-distances / (2 * sigma**2))

def mmd_rbf(X, Y, sigmas=[1, 2, 4, 8, 16]):
    K_XX = 0
    K_XY = 0
    K_YY = 0

    for sigma in sigmas:
        K_XX += rbf_kernel(X, X, sigma)
        K_XY += rbf_kernel(X, Y, sigma)
        K_YY += rbf_kernel(Y, Y, sigma)

    m = X.shape[0]
    n = Y.shape[0]

    # Unbiased estimator
    mmd2 = (K_XX.sum() - np.trace(K_XX)) / (m*(m-1)) \
         + (K_YY.sum() - np.trace(K_YY)) / (n*(n-1)) \
         - 2 * K_XY.mean()

    return mmd2


def compute_metrics(test_path, gen_path, subsample=False, n_sub=1000):
    #Load data (no pre-processing needed)
    test_adata = sc.read_h5ad(test_path)
    gen_adata  = sc.read_h5ad(gen_path)

    # Convert sparse to dense if needed
    X_test = test_adata.X.toarray() if hasattr(test_adata.X, "toarray") else test_adata.X
    X_gen  = gen_adata.X.toarray()  if hasattr(gen_adata.X,  "toarray") else gen_adata.X

    # PCA on test data (+ transform generated data)

    pca = PCA(n_components=30)
    PCs_test = pca.fit_transform(X_test)
    PCs_gen  = pca.transform(X_gen)

    #Subsample

    if subsample:
        if PCs_test.shape[0] > n_sub:
            PCs_test_sample = PCs_test[np.random.choice(PCs_test.shape[0], n_sub, replace=False)]
        else:
            PCs_test_sample = PCs_test

        if PCs_gen.shape[0] > n_sub:
            PCs_gen_sample = PCs_gen[np.random.choice(PCs_gen.shape[0], n_sub, replace=False)]
        else:
            PCs_gen_sample = PCs_gen
    else:
        PCs_test_sample = PCs_test
        PCs_gen_sample  = PCs_gen
    
    # Compute multivariate Wasserstein distance

    # Uniform weights (empirical distribution)
    a = np.ones(PCs_test_sample.shape[0]) / PCs_test_sample.shape[0]
    b = np.ones(PCs_gen_sample.shape[0]) / PCs_gen_sample.shape[0]

    # Ground distance matrix in 30D (Euclidean)
    M = ot.dist(PCs_test_sample, PCs_gen_sample, metric='euclidean')

    # Wasserstein-1 (EMD)

    W1 = ot.emd2(a, b, M)   # squared W1
    W1 = np.sqrt(W1)        # convert to W1


    # 2-Wasserstein (uses squared Euclidean distance)
    M2 = M ** 2
    W2_sq = ot.emd2(a, b, M2)   # already squared
    W2 = np.sqrt(W2_sq)

    # Compute MMD with RBF kernel

    mmd_value = mmd_rbf(PCs_test_sample, PCs_gen_sample)



    return W1, W2, mmd_value

def main():
    parser = argparse.ArgumentParser(description="Compute W1, W2, and MMD between test and generated data.")
    parser.add_argument("--test", type=str, required=True, help="Path to test .h5ad file")
    parser.add_argument("--gen",  type=str, required=True, help="Path to generated .h5ad file")
    parser.add_argument("--subsample", action="store_true", help="Enable subsampling")
    parser.add_argument("--n_sub", type=int, default=1000, help="Subsample size")

    args = parser.parse_args()

    W1, W2, MMD = compute_metrics(
        test_path=args.test,
        gen_path=args.gen,
        subsample=args.subsample,
        n_sub=args.n_sub
    )

    print("\nRESULTS:")
    print(f"Wasserstein-1 (W1):  {W1:.4f}")
    print(f"Wasserstein-2 (W2):  {W2:.4f}")
    print(f"MMD (multi-kernel):  {MMD:.6f}")



############DELETE ME AFTER
#### HOW TO CALL FROM JUPYTER:
#from compute_metrics import compute_metrics

#test_path = "../data/datasets/pbmc3k/pbmc3k_test.h5ad"
#gen_path  = "../data/datasets/pbmc3k/pbmc3k_train.h5ad"

#W1, W2, MMD = compute_metrics(test_path, gen_path, subsample=True)
#print(W1, W2, MMD)