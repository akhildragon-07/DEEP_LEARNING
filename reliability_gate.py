"""RMS-AV reliability gate.

Looks at the audio and video features of each clip and outputs two weights
(audio, video) that sum to 1. A missing modality always gets weight 0.
"""
import torch
import torch.nn as nn


class ReliabilityGate(nn.Module):
    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, a, v, has_audio, has_video):
        """
        a, v:                [B, D] audio / video features
        has_audio/has_video: [B] bool masks
        returns weights:     [B, 2]  (col 0 = audio, col 1 = video)
        """
        scores = self.net(torch.cat([a, v], dim=1))            # [B, 2]
        mask = torch.stack([has_audio, has_video], dim=1)      # [B, 2] bool
        scores = scores.masked_fill(~mask, float("-inf"))
        w = torch.softmax(scores, dim=1)
        # if both modalities are missing, softmax gives NaN -> use zeros
        w = torch.nan_to_num(w, nan=0.0)
        return w
