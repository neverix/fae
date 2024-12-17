#%%
from glob import glob
import numpy as np
from ml_dtypes import bfloat16
import matplotlib.pyplot as plt

datas = []
for i in glob("somewhere/td/*.npy"):
    datas.append(np.load(i))
data = np.concatenate(datas)
data.shape
#%%
plt.xscale("log")
# norm histogram
plt.hist(np.linalg.norm(data, axis=1), bins=1000, label="norm")
# norm away from mean histogram
plt.hist(np.linalg.norm(data - np.mean(data, axis=0), axis=1), bins=1000, label="norm away from mean")
plt.legend()
plt.show()
#%%
# PCA
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(data)
data_pca = pca.transform(data)
plt.scatter(data_pca[:, 0], data_pca[:, 1])
plt.xlabel("PC1")
plt.ylabel("PC2")
#%%
plt.xscale("log")
plt.hist(data_pca[:, 0], bins=1000)
plt.xlabel("PC1")

flagged = data_pca[:, 0] > 500
flagged.sum()
#%%
plt.plot(
    np.corrcoef(x=data_pca[flagged],
                rowvar=False)[0, 1:] ** 2)
plt.xlabel("Principal component")
plt.ylabel("R^2 with PC1 on samples affected by the outlier direction")
#%%
plt.plot(np.sort(data.std(axis=0))[::-1])
plt.xlabel("Rank of residual neuron by standard deviation")
plt.ylabel("Standard deviation")
plt.xscale("log")
plt.yscale("log")
#%%
plt.plot(data.max(axis=0) - data.min(axis=0))
plt.xlabel("Residual neuron #")
plt.ylabel("Amplitude of activation")
plt.ylim(0, 14000)
#%%
# Variance explained by PCA
pca = PCA()
pca.fit(data)
#%%
plt.plot(pca.explained_variance_ratio_)
plt.xscale("log")
plt.yscale("log")
# data is very low-rank!
# %%
# for comparison, let's look at data from an LM
llama_data = np.load("somewhere/llama.npz")["arr_0"]
pca = PCA()
pca.fit(llama_data)
#%%
plt.plot(pca.explained_variance_ratio_[1:])
plt.xscale("log")
# plt.yscale("log")
# %%
