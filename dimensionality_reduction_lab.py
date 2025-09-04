# %% [markdown]
# # Dimensionality Reduction Lab — PCA, LDA, KPCA
# This notebook follows Lab 5 instructions. Run cell by cell.
# Each section is clearly numbered: Part 1 (PCA), Part 2 (LDA), Part 3 (KPCA) with sub-parts.

# %% [markdown]
# ## Part 1: Principal Component Analysis (PCA)

# %% [markdown]
# ### 1.1 Load and Preprocess Wine Dataset

# %%
# Imports and setup
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_wine, make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
import warnings
warnings.filterwarnings('ignore')

# ensure figures folder exists
os.makedirs('figures', exist_ok=True)

# Load Wine dataset
wine = load_wine()
X = wine.data
y = wine.target
print("Data shape:", X.shape, "Labels:", np.unique(y))

# train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=0
)

# standardize features
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)
print("Standardized shapes:", X_train_std.shape, X_test_std.shape)

# %% [markdown]
# ### 1.2 PCA from Scratch (Covariance & Eigen decomposition)

# %%
# covariance matrix
cov_mat = np.cov(X_train_std.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
# sort eigenpairs
idx = np.argsort(eig_vals)[::-1]
eig_vals = np.real(eig_vals[idx])
eig_vecs = np.real(eig_vecs[:, idx])
# pick top-2 components
components = eig_vecs[:, :2]
# project data
X_train_pca_scratch = X_train_std.dot(components)
X_test_pca_scratch = X_test_std.dot(components)

# explained variance
var_exp = eig_vals / eig_vals.sum()
cum_var = np.cumsum(var_exp)

# plot explained variance
plt.figure(figsize=(6,4))
plt.bar(range(1, len(var_exp)+1), var_exp, alpha=0.5, align='center', label='Individual explained variance')
plt.step(range(1, len(cum_var)+1), cum_var, where='mid', label='Cumulative explained variance')
plt.xlabel('Principal components')
plt.ylabel('Explained variance ratio')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('figures/pca_explained_variance.png')
plt.show()

print("Top eigenvalues (first 6):", eig_vals[:6])
print("Cumulative variance (first 2 PCs):", cum_var[:2])

# %% [markdown]
# ### 1.3 PCA using scikit-learn

# %%
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

# %% [markdown]
# ### 1.4 Logistic Regression on PCA-transformed data

# %%
lr_pca = LogisticRegression(multi_class='ovr', solver='lbfgs', max_iter=2000, random_state=1)
lr_pca.fit(X_train_pca, y_train)

def plot_decision_regions(X, y, classifier, resolution=0.02, xlabel='x1', ylabel='x2', title=None):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    x1_min, x1_max = X[:,0].min() - 1, X[:,0].max() + 1
    x2_min, x2_max = X[:,1].min() - 1, X[:,1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.figure(figsize=(6,5))
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(X[y==cl,0], X[y==cl,1], alpha=0.8, c=[cmap(idx)], edgecolor='black', marker=markers[idx], label=f'class {cl}')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title: plt.title(title)
    plt.legend(loc='best')
    plt.tight_layout()

plot_decision_regions(X_train_pca, y_train, classifier=lr_pca, xlabel='PC1', ylabel='PC2',
                      title='Logistic Regression on PCA (sklearn) - Training')
plt.savefig('figures/pca_decision_regions.png')
plt.show()

print("PCA (sklearn) - Test accuracy:", lr_pca.score(X_test_pca, y_test))

# %% [markdown]
# ## Part 2: Linear Discriminant Analysis (LDA)

# %% [markdown]
# ### 2.1 LDA from Scratch (scatter matrices & eigen decomposition)

# %%
classes = np.unique(y_train)
d = X_train_std.shape[1]

mean_overall = np.mean(X_train_std, axis=0).reshape(d,1)
S_W = np.zeros((d,d), dtype=float)
S_B = np.zeros((d,d), dtype=float)

for c in classes:
    Xc = X_train_std[y_train==c]
    mean_vec = np.mean(Xc, axis=0).reshape(d,1)
    S_W += (Xc - mean_vec.ravel()).T.dot(Xc - mean_vec.ravel())
    n_c = Xc.shape[0]
    mean_diff = mean_vec - mean_overall
    S_B += n_c * mean_diff.dot(mean_diff.T)

eig_vals_lda, eig_vecs_lda = np.linalg.eig(np.linalg.pinv(S_W).dot(S_B))
idx_lda = np.argsort(np.real(eig_vals_lda))[::-1]
eig_vals_lda = np.real(eig_vals_lda[idx_lda])
eig_vecs_lda = np.real(eig_vecs_lda[:, idx_lda])

W = eig_vecs_lda[:, :2]
X_train_lda_scratch = X_train_std.dot(W)
X_test_lda_scratch = X_test_std.dot(W)

lr_lda_s = LogisticRegression(multi_class='ovr', solver='lbfgs', max_iter=2000, random_state=1)
lr_lda_s.fit(X_train_lda_scratch, y_train)

plot_decision_regions(X_train_lda_scratch, y_train, classifier=lr_lda_s, xlabel='LD1', ylabel='LD2',
                      title='LogReg on LDA (from scratch) - Training')
plt.savefig('figures/lda_decision_regions_scratch.png')
plt.show()

print("LDA (from scratch) - Test accuracy:", lr_lda_s.score(X_test_lda_scratch, y_test))

# %% [markdown]
# ### 2.2 LDA using scikit-learn

# %%
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)
X_test_lda = lda.transform(X_test_std)

lr_lda = LogisticRegression(multi_class='ovr', solver='lbfgs', max_iter=2000, random_state=1)
lr_lda.fit(X_train_lda, y_train)

plot_decision_regions(X_train_lda, y_train, classifier=lr_lda, xlabel='LD1', ylabel='LD2',
                      title='LogReg on LDA (sklearn) - Training')
plt.savefig('figures/lda_decision_regions_sklearn.png')
plt.show()

print("LDA (sklearn) - Test accuracy:", lr_lda.score(X_test_lda, y_test))

# %% [markdown]
# ## Part 3: Kernel PCA (KPCA)

# %% [markdown]
# ### 3.1 RBF Kernel PCA from Scratch

# %%
def rbf_kernel_pca(X, gamma=15, n_components=2):
    sq_dists = pdist(X, 'sqeuclidean')
    K = np.exp(-gamma * squareform(sq_dists))
    N = K.shape[0]
    one_n = np.ones((N,N)) / N
    K_center = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    eigvals, eigvecs = eigh(K_center)
    eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]
    alphas = eigvecs[:, :n_components]
    lambdas = eigvals[:n_components]
    alphas = alphas / np.sqrt(lambdas)
    return alphas, lambdas

# %% [markdown]
# ### 3.2 Half-Moon Dataset

# %%
X_m, y_m = make_moons(n_samples=200, noise=0.07, random_state=123)
for gamma in [0.01, 0.1, 1, 15, 100]:
    X_kpca, lambdas = rbf_kernel_pca(X_m, gamma=gamma, n_components=2)
    plt.figure(figsize=(5,4))
    plt.scatter(X_kpca[y_m==0,0], X_kpca[y_m==0,1], marker='^', alpha=0.7, label='class 0')
    plt.scatter(X_kpca[y_m==1,0], X_kpca[y_m==1,1], marker='o', alpha=0.7, label='class 1')
    plt.title(f'RBF KPCA (moons) — gamma={gamma}')
    plt.xlabel('PC1'); plt.ylabel('PC2')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'figures/kpca_moons_gamma_{gamma}.png')
    plt.show()

# %% [markdown]
# ### 3.3 Concentric Circles Dataset

# %%
X_c, y_c = make_circles(n_samples=600, factor=0.25, noise=0.06, random_state=123)
X_kpca_c, lambdas_c = rbf_kernel_pca(X_c, gamma=15, n_components=2)
plt.figure(figsize=(5,4))
plt.scatter(X_kpca_c[y_c==0,0], X_kpca_c[y_c==0,1], marker='^', alpha=0.6, label='class 0')
plt.scatter(X_kpca_c[y_c==1,0], X_kpca_c[y_c==1,1], marker='o', alpha=0.6, label='class 1')
plt.title('RBF KPCA (circles) — gamma=15')
plt.xlabel('PC1'); plt.ylabel('PC2')
plt.legend()
plt.tight_layout()
plt.savefig('figures/kpca_circles_gamma_15.png')
plt.show()

# %% [markdown]
# ### 3.4 KPCA using scikit-learn

# %%
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15, fit_inverse_transform=False)
X_kpca_sk = kpca.fit_transform(X_c)
plt.figure(figsize=(5,4))
plt.scatter(X_kpca_sk[y_c==0,0], X_kpca_sk[y_c==0,1], marker='^', alpha=0.6)
plt.scatter(X_kpca_sk[y_c==1,0], X_kpca_sk[y_c==1,1], marker='o', alpha=0.6)
plt.title('KernelPCA (sklearn) on circles (gamma=15)')
plt.tight_layout()
plt.savefig('figures/kpca_sklearn_circles.png')
plt.show()

# %% [markdown]
# ### End of Notebook — Guidance for Analysis
# - Explained variance (PCA): check cumulative variance plot.
# - PCA vs LDA: LDA uses class labels to maximize class separation.
# - KPCA gamma: small = underfitting, large = sensitive/overfit.
# - Compare classifier performance: check printed accuracies.
