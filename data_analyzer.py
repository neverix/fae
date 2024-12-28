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
