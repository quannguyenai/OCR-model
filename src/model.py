"""
CRNN Model for text recognition.
"""

import timm
import torch
import torch.nn as nn


class CRNN(nn.Module):
    """
    CRNN = ResNet34 (CNN) + BiGRU (RNN) + CTC
    
    Args:
        vocab_size: Number of characters (default: 37 for a-z, 0-9, blank)
        hidden_size: GRU hidden size
        n_layers: Number of GRU layers
        dropout: Dropout rate
    """
    
    def __init__(self, vocab_size=37, hidden_size=256, n_layers=3, dropout=0.2):
        super().__init__()
        
        # CNN backbone (ResNet34)
        backbone = timm.create_model("resnet34", in_chans=1, pretrained=True)
        modules = list(backbone.children())[:-2]
        modules.append(nn.AdaptiveAvgPool2d((1, None)))
        self.cnn = nn.Sequential(*modules)
        
        # Unfreeze last 3 layers
        for param in self.cnn[-3:].parameters():
            param.requires_grad = True
        
        # Map CNN features to sequence
        self.map = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # RNN (Bidirectional GRU)
        self.rnn = nn.GRU(
            512, hidden_size, n_layers,
            bidirectional=True, batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        
        self.norm = nn.LayerNorm(hidden_size * 2)
        
        # Output layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, vocab_size),
            nn.LogSoftmax(dim=2)
        )

    def forward(self, x):
        # x: (batch, 1, H, W)
        x = self.cnn(x)                    # (batch, 512, 1, seq_len)
        x = x.squeeze(2).permute(0, 2, 1)  # (batch, seq_len, 512)
        x = self.map(x)
        x, _ = self.rnn(x)
        x = self.norm(x)
        x = self.fc(x)
        return x.permute(1, 0, 2)          # (seq_len, batch, vocab) for CTC
