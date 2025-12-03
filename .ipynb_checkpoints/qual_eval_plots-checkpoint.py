#Qualitative evaluation plot functions
#import adata
import scanpy as sc
import numpy as np
from matplotlib import pyplot as plt

#Input is paths

def load_data(test_data_path, gen_data_path):
    #adata_PATH = "/dtu/blackhole/06/213542/paperdata/pbmc3k_test.h5ad"
    adata_real = sc.read_h5ad(test_data_path)
    #adata_PATH_generated = "/dtu/blackhole/06/213542/paperdata/pbmc3k_train.h5ad"
    adata_generated = sc.read_h5ad(gen_data_path)

    X_real = adata_real.X # real counts
    X_gen = adata_generated.X #generated counts

    X_real = np.asarray(adata_real.X.todense() if hasattr(adata_real.X, "todense") else adata_real.X)
    X_gen = np.asarray(adata_generated.X.todense() if hasattr(adata_generated.X, "todense") else adata_generated.X)

    return X_real, X_gen


#SPARSITY
def sparsity_plot(test_data_path, gen_data_path, save_path=""): #the default is that we don't save it, we just plot it
    X_real, X_gen = load_data(test_data_path, gen_data_path)
    
    zeros_real = (X_real == 0).sum(axis=1)
    zeros_fake = (X_gen == 0).sum(axis=1)
    plt.hist(zeros_real, bins=50, alpha=0.5, label="Real")
    plt.hist(zeros_fake, bins=50, alpha=0.5, label="Generated")
    plt.xlabel("# Zero Counts per Cell")
    plt.ylabel("Frequency")
    plt.legend()
    
    # Save if a path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')  # saves figure
        print(f"Figure saved in {save_path}")
    
    plt.show()
    
    return


#OVER-DISPRESION
def overdispersion_plot(test_data_path, gen_data_path, save_path=""): #the default is that we don't save it, we just plot it
    X_real, X_gen = load_data(test_data_path, gen_data_path)
    gene_mean_real = X_real.mean(axis=0)
    gene_var_real  = X_real.var(axis=0)
    
    gene_mean_fake = X_gen.mean(axis=0)
    gene_var_fake  = X_gen.var(axis=0)
    
    plt.scatter(gene_mean_real, gene_var_real, alpha=0.4, label="Real", s=10)
    plt.scatter(gene_mean_fake, gene_var_fake, alpha=0.4, label="Gerated", s=10)
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("Gene Mean"); plt.ylabel("Gene Variance")
    plt.legend()

    # Save if a path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')  # saves figure
        print(f"Figure saved in {save_path}")
    
    plt.show()
    
    return

#DISCRETNESS
def discreteness_plot(test_data_path, gen_data_path, save_path=""): #the default is that we don't save it, we just plot it
    X_real, X_gen = load_data(test_data_path, gen_data_path)
    #print("Real unique values:", np.unique(X_real[:50]))
    #print("Generated unique values:", np.unique(X_gen[:50]))
    
    plt.hist(X_gen.flatten(), bins=50, range=(0.01, X_gen.max()) )
    plt.title("Generated Count Distribution")
    # Save if a path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')  # saves figure
        print(f"Figure saved in {save_path}")
    
    plt.show()
    
    return

#FIG 2a

from sklearn.linear_model import LinearRegression

def preprocess_log_gexp(X):
    # X: counts matrix
    # Step 1: normalize each cell to 10k counts (CPM-like)
    norm = X / X.sum(axis=1, keepdims=True) * 1e4
    
    # Step 2: log-transform
    log_gexp = np.log1p(norm)
    return log_gexp

def add_trendline(x, y, color):
    model = LinearRegression().fit(x.reshape(-1, 1), y)
    x_line = np.linspace(0, x.max(), 100)
    y_line = model.predict(x_line.reshape(-1, 1))
    plt.plot(x_line, y_line, color=color, linewidth=3, alpha=0.9)

def hlca_plot(test_data_path, gen_data_path, save_path=""): #the default is that we don't save it, we just plot it
    X_real, X_gen = load_data(test_data_path, gen_data_path)
    
    log_real = preprocess_log_gexp(X_real)
    log_cfgen = preprocess_log_gexp(X_gen)
    
    mean_real = log_real.mean(axis=0)
    var_real  = log_real.var(axis=0)
    
    mean_cfgen = log_cfgen.mean(axis=0)
    var_cfgen  = log_cfgen.var(axis=0)



    plt.figure(figsize=(7,6))
    
    # scatter (low alpha, large cloud)
    plt.scatter(mean_real,  var_real,  s=10, alpha=0.2, color="#1f77b4")
    plt.scatter(mean_cfgen, var_cfgen, s=10, alpha=0.2, color="#ff7f0e")
    
    # axis formatting (important!)
    plt.xlim(0, 0.22)
    plt.ylim(0, 0.52)
    
    plt.xlabel("Gene mean (log-GEXP)", fontsize=14)
    plt.ylabel("Gene variance (log-GEXP)", fontsize=14)
    plt.title("HLCA", fontsize=18)
    
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.tight_layout()
    # Save if a path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')  # saves figure
        print(f"Figure saved in {save_path}")
    
    plt.show()
    
    return

#FIG 2B
def sparsity_density_plot(test_data_path, gen_data_path, save_path=""): #the default is that we don't save it, we just plot it
    X_real, X_gen = load_data(test_data_path, gen_data_path)

    # --- Make sure matrices are dense numpy arrays ---
    if hasattr(X_real, "toarray"):
        X_real = X_real.toarray()
    else:
        X_real = np.asarray(X_real)
    
    if hasattr(X_gen, "toarray"):
        X_gen = X_gen.toarray()
    else:
        X_gen = np.asarray(X_gen)
    
    print("Real shape:", X_real.shape)
    print("Generated shape:", X_gen.shape)
    
    # --- 1. Count zeros per cell ---
    zeros_real = (X_real == 0).sum(axis=1)   # one number per cell
    zeros_gen  = (X_gen  == 0).sum(axis=1)
    
    # (optional) fraction of zeros instead of counts
    frac_zeros_real = zeros_real / X_real.shape[1]
    frac_zeros_gen  = zeros_gen  / X_gen.shape[1]
    
    # --- 2. Plot like Fig 2b: histogram of zeros per cell ---
    plt.figure(figsize=(6,5))
    
    plt.hist(zeros_real, bins=50, alpha=0.5, density=True, label="Real")
    plt.hist(zeros_gen,  bins=50, alpha=0.5, density=True, label="Generated")
    
    plt.xlabel("# zero counts per cell", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.title("Sparsity: zeros per cell", fontsize=16)
    plt.legend()
    plt.tight_layout()
    
    # Save if a path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')  # saves figure
        print(f"Figure saved in {save_path}")
    
    plt.show()
    
    return