import torch
import torch.nn as nn


class ConvolutionalAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            # 1 × 64 × 64
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            # 32 × 32 × 32
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            # 64 × 16 × 16
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            # 128 × 8 × 8
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.ReLU()
        )


        # Decoder
        self.decoder = nn.Sequential(
            # 256 × 4 × 4
            nn.ConvTranspose2d(
                256,
                128,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.ReLU(),

            # 128 × 8 × 8
            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.ReLU(),

            # 64 × 16 × 16
            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.ReLU(),

            # 32 × 32 × 32
            nn.ConvTranspose2d(
                32,
                1,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),

            # Output in [0,1]
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed