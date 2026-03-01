import torch.nn as nn

class DescriptorEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        return self.encoder(x)
