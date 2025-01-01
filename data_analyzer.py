#%%
import os
os.environ["JAX_PLATFORMS"] = "cpu"
from src.fae.quant_loading import restore_array
sae_mid = restore_array(os.path.abspath("somewhere/sae_mid/2000/default"))
#%%
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

text_data = np.concatenate(list(map(np.load, glob("somewhere/td/*.npy"))))
image_double_data = np.concatenate(list(map(np.load, glob("somewhere/isd/*.npy"))))
image_single_data = np.concatenate(list(map(np.load, glob("somewhere/idd/*.npy"))))
#%%
plt.xscale("log")
# norm histogram
plt.hist(np.linalg.norm(text_data, axis=1), bins=1000, label="norm")
# norm away from mean histogram
plt.hist(np.linalg.norm(text_data - np.mean(text_data, axis=0), axis=1), bins=1000, label="norm away from mean")
plt.legend()
plt.show()
#%%
# PCA
from sklearn.decomposition import PCA
text_pca = PCA()
text_pca.fit(text_data)
text_data_pca = text_pca.transform(text_data)
#%%
plt.scatter(text_data_pca[:, 0], text_data_pca[:, 1], c=(np.arange(len(text_data_pca)) % 512) / 512., s=0.01)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar()
#%%
# plt.xscale("log")
plt.hist(text_data_pca[:, 0], bins=1000)
plt.xlabel("PC1")

flagged = text_data_pca[:, 0] > 500
flagged.sum()
#%%
plt.plot(
    np.corrcoef(x=text_data_pca[flagged],
                rowvar=False)[0, 1:] ** 2)
plt.xlabel("Principal component")
plt.ylabel("R^2 with PC1 on samples affected by the outlier direction")
#%%
plt.plot(np.sort(text_data.std(axis=0))[::-1])
plt.xlabel("Rank of residual neuron by standard deviation")
plt.ylabel("Standard deviation")
plt.xscale("log")
plt.yscale("log")
#%%
plt.plot(text_data.max(axis=0) - text_data.min(axis=0))
plt.plot(text_data.std(axis=0))
plt.xlabel("Residual neuron #")
plt.ylabel("Amplitude of activation")
# plt.ylim(0, 14000)
#%%
# Variance explained by PCA
double_pca = PCA()
double_pca.fit(image_double_data)
single_pca = PCA()
single_pca.fit(image_single_data)
#%%
plt.plot(double_pca.explained_variance_ratio_, label="double")
plt.plot(single_pca.explained_variance_ratio_, label="single")
plt.xlabel("Principal component")
plt.ylabel("Variance explained")
plt.legend()
plt.xscale("log")
plt.yscale("log")
# data is very low-rank!
# %%
# visualize double/single neuron amplitudes
plt.plot(image_single_data.max(axis=0) - image_single_data.min(axis=0), label="single")
plt.plot(image_double_data.max(axis=0) - image_double_data.min(axis=0), label="double")
plt.xlabel("Residual neuron #")
plt.ylabel("Amplitude of activation")
plt.legend()
plt.show()
#%%
# Same data, but sorted by double neuron std
double_sort_order = np.argsort(image_double_data.std(axis=0))[::-1]
single_sort_order = np.argsort(image_single_data.std(axis=0))[::-1]
plt.plot(image_double_data.std(axis=0)[double_sort_order], label="double")
# plt.plot(image_single_data.std(axis=0)[double_sort_order], label="single")
plt.plot(image_single_data.std(axis=0)[single_sort_order], label="single")
plt.xlabel("Index sorted by neuron std")
plt.ylabel("Standard deviation")
plt.yscale("log")
plt.xscale("log")
plt.legend()
plt.show()
# %%
plt.scatter(image_single_data.std(axis=0), text_data.std(axis=0), s=10)
plt.ylim(0, 1000)
#%%
double_pca_data = double_pca.transform(image_double_data)
plt.scatter(double_pca_data[:, 0], double_pca_data[:, 1], s=10)
plt.xlabel("PC1")
plt.ylabel("PC2")
#%%
double_norms = np.linalg.norm(image_double_data, axis=1)
plt.hist(double_norms, bins=1000)
plt.show()
#%%
plt.plot(np.sort(double_pca.explained_variance_ratio_)[::-1])
plt.plot(np.sort(single_pca.explained_variance_ratio_)[::-1])
plt.xscale("log")
plt.yscale("log")
plt.show()
#%%
plt.plot(np.sort(image_double_data.var(axis=0))[::-1])
plt.plot(np.sort(image_single_data.var(axis=0))[::-1])
plt.plot(np.sort(text_data.var(axis=0))[::-1])
plt.hlines([8192], 0, text_data.shape[1])

plt.xscale("log")
plt.yscale("log")
plt.show()
#%%
plt.plot(np.sort(image_double_data.max(axis=0) - image_double_data.min(axis=0))[::-1])
plt.plot(np.sort(image_single_data.max(axis=0) - image_single_data.min(axis=0))[::-1])
plt.plot(np.sort(text_data.var(axis=0))[::-1])
plt.hlines([8192], 0, text_data.shape[1])

plt.xscale("log")
plt.yscale("log")
plt.show()
#%%
double_mean = image_double_data.mean(axis=0)
double_std = image_double_data.std(axis=0)
double_deanisotropized = (image_double_data - double_mean) / double_std
# plt.plot(np.sort(double_deanisotropized.var(axis=0))[::-1])
#%%
plt.plot(np.sort(double_deanisotropized.max(axis=0) - double_deanisotropized.min(axis=0))[::-1], label="deaniso")
plt.plot(np.sort(image_double_data.max(axis=0) - image_double_data.min(axis=0))[::-1], label="og")
plt.xscale("log")
plt.yscale("log")
plt.legend()
#%%
from sklearn.decomposition import PCA
pca_deaniso = PCA()
pca_deaniso.fit(double_deanisotropized)
pca_double = PCA()
pca_double.fit(image_double_data)
plt.plot(pca_deaniso.explained_variance_ratio_, label="deaniso")
plt.plot(pca_double.explained_variance_ratio_, label="og")
plt.xlabel("Principal component")
plt.ylabel("Variance explained")
plt.xscale("log")
plt.yscale("log")
plt.legend()
#%%
double_decentered = image_double_data - image_double_data.mean(axis=0)
double_cov = double_decentered.T @ double_decentered
double_cov /= len(image_double_data)
double_cov_subsampled = double_cov
for _ in range(7):
    double_cov_subsampled = np.maximum(
        np.maximum(
        double_cov_subsampled[::2, ::2],
        double_cov_subsampled[1::2, ::2],
        ),
        np.maximum(
        double_cov_subsampled[::2, 1::2],
        double_cov_subsampled[1::2, 1::2]
        )
    )
plt.imshow(double_cov_subsampled)
plt.colorbar()
#%%
# double_pca_trans = np.linalg.eigh(double_cov)
u, s, vt = np.linalg.svd(double_decentered)
#%%
# double_v = double_pca_trans.eigenvectors
# double_s = double_pca_trans.eigenvalues ** 0.5
double_v = vt.T
double_s = s
double_pcaed = double_decentered @ double_v
(np.mean(np.square(double_pcaed @ double_v.T - double_decentered)),
np.mean(np.square(double_decentered)), np.var(double_pcaed[:, 0]))
#%%
def amp(x, axis):
    return np.max(x, axis=axis) - np.min(x, axis=axis)
plt.plot(np.sort(amp(double_pcaed, axis=0))[::-1], label="post-PCA")
plt.plot(np.sort(amp(double_deanisotropized, axis=0))[::-1], label="pre-PCA")
plt.xscale("log")
# plt.yscale("log")
plt.legend()
plt.xlabel("Neuron # (or PC #)")
plt.ylabel("Amplitude after standardization")
#%%
# http://neilsloane.com/hadamard/
had_12 = """+-----------
++-+---+++-+
+++-+---+++-
+-++-+---+++
++-++-+---++
+++-++-+---+
++++-++-+---
+-+++-++-+--
+--+++-++-+-
+---+++-++-+
++---+++-++-
+-+---+++-++"""
had_12 = np.array([[1 if c == "+" else -1 for c in row] for row in had_12.split("\n") if row])
def hadamard_matrix(n):
    if n == 1:
        return np.array([[1]])
    elif n == 12:
        return had_12
    else:
        h = hadamard_matrix(n // 2)
        return np.block([[h, h], [h, -h]])
hada = hadamard_matrix(3072)
d = image_double_data.shape[1]
hada = hada[:d, :d] / (d ** 0.5)
double_recon = double_decentered @ hada @ hada.T
np.mean(np.square(double_recon - double_decentered))
# %%
# double_hada = double_decentered @ hada
double_hada = image_double_data @ hada
# double_hada = double_deanisotropized @ hada
double_hada = double_hada - double_hada.mean(axis=0)
double_hada = double_hada / np.std(double_hada, axis=0)
plt.plot(np.sort(amp(double_hada, axis=0))[::-1], label="post-Hadamard")
plt.plot(np.sort(amp(double_pcaed, axis=0))[::-1], label="post-PCA")
plt.plot(np.sort(amp(double_deanisotropized, axis=0))[::-1], label="pre-everything")
plt.xscale("log")
plt.yscale("log")
plt.legend()
# %%
