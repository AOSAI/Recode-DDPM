import torch
import matplotlib.pyplot as plt

from ddpm.nn import timestep_embedding

emb = timestep_embedding(torch.arange(100), dim=64)
plt.imshow(emb.T, aspect='auto', cmap='magma')
plt.xlabel("Time step t")
plt.ylabel("Embedding dim")
plt.colorbar()
plt.title("Sinusoidal Timestep Embedding")
plt.show()
