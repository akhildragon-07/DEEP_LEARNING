"""Audio-visual deepfake detector (Member 2 RMS-AV Integration).

Modes:
  audio  : audio encoder + audio head only
  video  : video encoder + video head only
  fusion : concat(audio, video) -> classifier (standard fusion)
  rmsav  : reliability-gated features -> classifier (RMS-AV)

Batch dict (agreed interface):
  video     [B, T, 3, H, W]
  audio     [B, 1, n_mels, time]
  has_audio [B] bool
  has_video [B] bool

Output dict:
  logit_av, logit_audio, logit_video : [B]   (raw logits, fake = 1)
  gate : [B, 2] (rmsav only, else None)
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torchvision

from reliability_gate import ReliabilityGate

# Backwards compatibility with legacy PG-ABN codebase
try:
    from model_pgabn import PGABN, build_pgabn_model
except ImportError:
    PGABN = None
    build_pgabn_model = None

MODES = ("audio", "video", "fusion", "rmsav")


class VideoEncoder(nn.Module):
    """ResNet18 on each frame, averaged over time."""

    def __init__(self, dim: int = 256, pretrained: bool = False, freeze: bool = False):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        r = torchvision.models.resnet18(weights=weights)
        r.fc = nn.Identity()
        self.backbone = r
        self.proj = nn.Linear(512, dim)
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, 3, H, W] -> returns [B, dim]"""
        B, T = x.shape[:2]
        f = self.backbone(x.flatten(0, 1))      # [B*T, 512]
        f = f.view(B, T, -1).mean(dim=1)        # [B, 512]
        return self.proj(f)                     # [B, dim]


class AudioEncoder(nn.Module):
    """Small CNN on a mel spectrogram."""

    def __init__(self, dim: int = 256):
        super().__init__()

        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1),
                nn.BatchNorm2d(o),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )

        self.cnn = nn.Sequential(block(1, 16), block(16, 32), block(32, 64))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(64, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 1, n_mels, time] -> returns [B, dim]"""
        f = self.pool(self.cnn(x)).flatten(1)   # [B, 64]
        return self.proj(f)                     # [B, dim]


def _head(dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(dim, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 1),
    )


class AVDetector(nn.Module):
    def __init__(
        self,
        mode: str = "rmsav",
        dim: int = 256,
        pretrained: bool = False,
        freeze_video: bool = False,
    ):
        super().__init__()
        assert mode in MODES, f"mode must be one of {MODES}"
        self.mode = mode
        self.dim = dim

        # Build only what each mode needs, so parameter counts are honest
        self.video_enc = VideoEncoder(dim, pretrained, freeze_video) if mode != "audio" else None
        self.audio_enc = AudioEncoder(dim) if mode != "video" else None
        self.audio_head = _head(dim) if mode != "video" else None
        self.video_head = _head(dim) if mode != "audio" else None

        if mode == "rmsav":
            self.gate = ReliabilityGate(dim)
        if mode in ("fusion", "rmsav"):
            self.fusion_head = _head(2 * dim)

    def forward(
        self,
        batch: Optional[Dict[str, torch.Tensor]] = None,
        video: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        has_video: Optional[torch.Tensor] = None,
        has_audio: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Accepts either a batch dict with keys ('video', 'audio', 'has_video', 'has_audio')
        or explicit keyword arguments.
        """
        if batch is not None:
            v_input = batch.get("video")
            a_input = batch.get("audio")
            has_v = batch["has_video"].bool()
            has_a = batch["has_audio"].bool()
        else:
            v_input = video
            a_input = audio
            has_v = has_video.bool() if has_video is not None else torch.tensor([v_input is not None])
            has_a = has_audio.bool() if has_audio is not None else torch.tensor([a_input is not None])

        B = has_a.shape[0]
        zeros = torch.zeros(B, self.dim, device=has_a.device)

        # Encode, then zero out features of missing modalities
        if self.audio_enc is not None and a_input is not None:
            a = self.audio_enc(a_input)
        else:
            a = zeros

        if self.video_enc is not None and v_input is not None:
            v = self.video_enc(v_input)
        else:
            v = zeros

        a = a * has_a.unsqueeze(1).float()
        v = v * has_v.unsqueeze(1).float()

        zero_logit = torch.zeros(B, device=has_a.device)
        logit_a = self.audio_head(a).squeeze(1) if self.audio_head is not None else zero_logit
        logit_v = self.video_head(v).squeeze(1) if self.video_head is not None else zero_logit

        gate = None
        if self.mode == "audio":
            logit_av = logit_a
        elif self.mode == "video":
            logit_av = logit_v
        else:
            if self.mode == "rmsav":
                gate = self.gate(a, v, has_a, has_v)       # [B, 2]
                a = a * gate[:, 0:1]
                v = v * gate[:, 1:2]
            logit_av = self.fusion_head(torch.cat([a, v], dim=1)).squeeze(1)

        return {
            "logit_av": logit_av,
            "logit_audio": logit_a,
            "logit_video": logit_v,
            "gate": gate,
        }


def build_model(mode: str = "rmsav", cfg: Optional[Dict] = None) -> AVDetector:
    cfg = cfg or {}
    mode_clean = mode.lower().strip()
    if mode_clean in ["standard_fusion", "standard_av", "fusion"]:
        mode_clean = "fusion"
    elif mode_clean in ["video_only", "video"]:
        mode_clean = "video"
    elif mode_clean in ["audio_only", "audio"]:
        mode_clean = "audio"
    elif mode_clean in ["rms_av", "rmsav"]:
        mode_clean = "rmsav"

    return AVDetector(
        mode=mode_clean,
        dim=cfg.get("dim", 256),
        pretrained=cfg.get("pretrained", False),
        freeze_video=cfg.get("freeze_video", False),
    )


def count_params(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
