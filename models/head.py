import torch
import torch.nn as nn


class FusionHead(torch.nn.Module):
    def __init__(self, embed_size=1024 * 16):
        super(FusionHead, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_size, 512 * 16),
            nn.LeakyReLU(0.01),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512 * 16, 256 * 16),
            nn.LeakyReLU(0.01),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256 * 16, 1)
        )

    def forward(self, emb):
        out = self.fc(emb)
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0.0)



