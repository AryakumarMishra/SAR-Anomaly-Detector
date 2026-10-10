import torch
from models.cae import ConvolutionalAutoencoder


model = ConvolutionalAutoencoder()

x = torch.randn(4, 1, 64, 64)

y = model(x)

print("Input shape:", x.shape)
print("Output shape:", y.shape)
print("Parameters:", sum(p.numel() for p in model.parameters()))