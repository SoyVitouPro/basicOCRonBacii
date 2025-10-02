import torch
import torch.nn as nn
import torch.nn.functional as F

class CRNN(nn.Module):
    def __init__(self, img_h, num_classes, hidden_size=256, num_lstm_layers=2):
        super().__init__()
        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),  # (C, H, W) -> (64, H, W)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),          # H: 32 -> 16,  W: /2

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),          # H: 16 -> 8,   W: /2

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),

            # Key line: collapse height to exactly 1, keep width unchanged
            nn.AdaptiveAvgPool2d((1, None)),
        )

        # After adaptive pooling, features will be [B, 256, 1, W’]
        # So the per-timestep feature dim is 256 (not dependent on img_h).
        self.rnn = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            bidirectional=True,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # x: [B, 1, H, W]
        features = self.cnn(x)                 # [B, 256, 1, W’]
        b, c, h, w = features.size()           # h will be 1
        features = features.squeeze(2)         # [B, 256, W’]
        features = features.permute(0, 2, 1)   # [B, W’, 256]  (time-major in dim 1)

        out, _ = self.rnn(features)            # [B, W’, 2*hidden]
        out = self.fc(out)                     # [B, W’, num_classes]
        out = out.permute(1, 0, 2)             # [W’, B, num_classes] for CTCLoss
        return out
