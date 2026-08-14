"""
Exact replication of the 13-layer CNN from Acharya et al. (2018).
"""

import torch
import torch.nn as nn
from config import CNN_LAYERS, FC_SIZES, LEAKY_RELU_SLOPE, SEGMENT_LENGTH


class EEG_CNN(nn.Module):
    """13-layer deep CNN for 3-class EEG classification."""

    def __init__(self):
        super().__init__()

        layers = []
        in_channels = 1

        for layer_type, out_channels, kernel_size, stride in CNN_LAYERS:
            if layer_type == "conv":
                layers.append(
                    nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride)
                )
                layers.append(nn.LeakyReLU(negative_slope=LEAKY_RELU_SLOPE))
                in_channels = out_channels
            elif layer_type == "pool":
                layers.append(nn.MaxPool1d(kernel_size, stride=stride))

        self.features = nn.Sequential(*layers)

        # Compute flattened size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, SEGMENT_LENGTH)
            out = self.features(dummy)
            flat_size = out.view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(flat_size, FC_SIZES[0]),
            nn.LeakyReLU(negative_slope=LEAKY_RELU_SLOPE),
            nn.Linear(FC_SIZES[0], FC_SIZES[1]),
            nn.LeakyReLU(negative_slope=LEAKY_RELU_SLOPE),
            nn.Linear(FC_SIZES[1], FC_SIZES[2]),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier/Glorot initialization for conv and linear layers."""
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = EEG_CNN()
    print(model)
    print(f"\nTotal trainable parameters: {count_parameters(model):,}")
    x = torch.randn(3, 1, SEGMENT_LENGTH)
    out = model(x)
    print(f"Input: {x.shape} -> Output: {out.shape}")
