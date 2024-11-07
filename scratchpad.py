#%%
import ml_dtypes
import torch
torch.set_grad_enabled(False)
text_encodings = torch.load("somewhere/res.pt", map_location=torch.device("cpu"), weights_only=True)
t5_emb, clip_emb = (x.detach().cpu().float().numpy() for x in text_encodings[:2])
# %%
import numpy as np
t5_emb_2 = np.load("somewhere/quantized_states.npy")[:, 0]
print(t5_emb.shape, t5_emb_2.shape)
print(
    np.mean(np.abs(t5_emb - t5_emb_2)),
    np.mean(np.abs(t5_emb)),
    np.mean(np.abs(t5_emb_2)),
)
print(
    np.mean(np.abs(t5_emb - t5_emb_2), -1)[0],
    np.mean(np.abs(t5_emb), -1)[0],
    np.mean(np.abs(t5_emb_2), -1)[0],
)

