import json, os, math
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms
from utils import save_experiment, save_checkpoint, load_experiment, visualize_images, visualize_attention

exp_name = 'vit-with-10-epochs'
config, model, train_losses, test_losses, accuracies = load_experiment(f"experiments/{exp_name}/")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.plot(train_losses, label="Train loss")
ax1.plot(test_losses, label="Test loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()
ax2.plot(accuracies)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
plt.show()

visualize_attention(model, "attention.png", device)
