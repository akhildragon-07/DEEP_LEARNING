r"""
LAV-DF Multimodal Deepfake Detection Architectures (RMS-AV 2.0)
--------------------------------------------------------------
Implements four model configurations:
1. Audio-only baseline (`mode="audio"`): 4-layer 2D CNN acoustic encoder + classification head.
2. Video-only baseline (`mode="video"`): ResNet18 visual encoder + Temporal Saliency Aggregator + classification head.
3. Standard Fusion (`mode="fusion"`): Feature concatenation [v; a] + joint MLP classification head.
4. RMS-AV (`mode="rmsav"`): Reliability- and Manipulation-Aware Selective Audio-Visual Fusion
   - Temporal Saliency Aggregation (TSA) over visual frames.
   - Cross-Modal Forensic Contrastive Incoherence (CM-FCI) Module.
   - Dynamic Incoherence-Guided Reliability Gate with FP16 validity masking.
   - Auxiliary unimodal authenticity heads for visual and acoustic streams.
   - Selective weighted fusion robust to asymmetric forgery and missing streams.
"""

import math
from typing import Dict, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class AudioEncoder(nn.Module):
    """
    Acoustic feature extractor for 80-channel log-mel spectrograms.
    Input: [B, 1, 80, T]
    Output: Latent embedding [B, dim]
    """
    def __init__(self, in_channels: int = 1, dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [B, 32, 40, T/2]

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [B, 64, 20, T/4]

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [B, 128, 10, T/8]

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # [B, 256, 1, 1]
        )
        self.fc = nn.Linear(256, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = torch.flatten(h, 1)
        out = self.fc(h)
        return out


class TemporalSaliencyAggregator(nn.Module):
    """
    Temporal Saliency Attention Aggregator across video frames.
    Replaces static uniform pooling with bidirectional temporal context
    and attention saliency pooling to detect transient deepfake flicker
    and inter-frame boundary splicing.
    """
    def __init__(self, dim: int = 256):
        super().__init__()
        self.gru = nn.GRU(
            input_size=dim,
            hidden_size=dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.Tanh(),
            nn.Linear(dim // 4, 1),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input: [B, T, dim]
        Output:
          pooled: [B, dim]
          weights: [B, T]
        """
        # Bidirectional temporal dynamics
        h_temp, _ = self.gru(x)  # [B, T, dim]
        # Residual skip connection
        fused = self.norm(x + h_temp)
        # Frame saliency scores
        scores = self.attention(fused).squeeze(-1)  # [B, T]
        alpha = F.softmax(scores, dim=-1)  # [B, T]
        # Attention-weighted pooling
        pooled = torch.bmm(alpha.unsqueeze(1), fused).squeeze(1)  # [B, dim]
        return pooled, alpha


class VisualEncoder(nn.Module):
    """
    Visual feature extractor based on ResNet18 with Temporal Saliency Aggregation.
    Input: [B, T, 3, H, W]
    Output: Latent embedding [B, dim], Temporal Attention Weights [B, T]
    """
    def __init__(self, dim: int = 256, pretrained: bool = False, use_tsa: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)

        # Exclude the final classification fc layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # outputs [B*T, 512, 1, 1]
        self.fc = nn.Linear(512, dim)
        self.use_tsa = use_tsa
        if use_tsa:
            self.tsa = TemporalSaliencyAggregator(dim=dim)
        else:
            self.tsa = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        b, t, c, h, w = x.shape
        x_flat = x.view(b * t, c, h, w)
        feats = self.backbone(x_flat)  # [B*T, 512, 1, 1]
        feats = torch.flatten(feats, 1)  # [B*T, 512]
        proj = self.fc(feats)            # [B*T, dim]
        proj = proj.view(b, t, -1)       # [B, T, dim]

        if self.use_tsa and self.tsa is not None:
            pooled, alpha = self.tsa(proj)
            return pooled, alpha
        else:
            pooled = proj.mean(dim=1)
            return pooled, None


class CrossModalIncoherenceProjector(nn.Module):
    """
    Cross-Modal Forensic Contrastive Incoherence (CM-FCI) Module.
    Projects acoustic and visual latent representations into a shared unit hypersphere
    to explicitly measure and penalize cross-modal physical/identity dissonance.
    """
    def __init__(self, dim: int = 256, proj_dim: int = 128):
        super().__init__()
        self.proj_v = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, proj_dim),
        )
        self.proj_a = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, proj_dim),
        )

    def forward(
        self, feat_v: torch.Tensor, feat_a: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          z_v: Normalized visual projection [B, proj_dim]
          z_a: Normalized acoustic projection [B, proj_dim]
          incoherence: Normalized discordance distance [B, 1] in [0, 1]
        """
        z_v = F.normalize(self.proj_v(feat_v), p=2, dim=-1)
        z_a = F.normalize(self.proj_a(feat_a), p=2, dim=-1)
        # Cosine similarity in [-1, 1]
        cos_sim = (z_v * z_a).sum(dim=-1, keepdim=True)  # [B, 1]
        # Incoherence metric: 0.0 = perfectly concordant (real), 1.0 = diametrically opposed (fake)
        incoherence = torch.clamp((1.0 - cos_sim) / 2.0, 0.0, 1.0)
        return z_v, z_a, incoherence

    def compute_fci_loss(
        self, z_v: torch.Tensor, z_a: torch.Tensor, labels: torch.Tensor, margin: float = 0.20
    ) -> torch.Tensor:
        """
        Supervised Contrastive Incoherence Loss.
        Authentic samples (y=0) are pulled together (cos -> 1).
        Deepfake samples (y=1) are pushed apart beyond margin m (cos <= margin).
        """
        cos_sim = (z_v * z_a).sum(dim=-1)  # [B]
        # For real (0): minimize (1 - cos_sim)
        loss_real = 1.0 - cos_sim
        # For fake (1): penalize if cos_sim > margin
        loss_fake = F.relu(cos_sim - margin)
        # Composite loss
        y = labels.float()
        loss = (1.0 - y) * loss_real + y * loss_fake
        return loss.mean()


class ReliabilityGate(nn.Module):
    """
    Dynamic Reliability Gate with Incoherence Guidance.
    Calculates dynamic attention weights [g_a, g_v] with validity masking
    and cross-modal dissonance modulation.
    """
    def __init__(self, dim: int = 256):
        super().__init__()
        # Input: [feat_a; feat_v; incoherence_metric] -> dim * 2 + 1
        self.gate_net = nn.Sequential(
            nn.Linear(dim * 2 + 1, dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 2, 2),
        )
        # Initialize final gate layer to zero so gates start at neutral [0.5, 0.5] prior
        nn.init.zeros_(self.gate_net[2].weight)
        nn.init.zeros_(self.gate_net[2].bias)

    def forward(
        self,
        feat_a: torch.Tensor,
        feat_v: torch.Tensor,
        incoherence: Optional[torch.Tensor] = None,
        has_audio: Optional[torch.Tensor] = None,
        has_video: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns normalized gate weights [B, 2] corresponding to [g_a, g_v].
        """
        if incoherence is None:
            incoherence = torch.zeros(feat_a.shape[0], 1, device=feat_a.device)

        combined = torch.cat([feat_a, feat_v, incoherence], dim=-1)
        raw_scores = self.gate_net(combined)  # [B, 2]

        if has_audio is not None and has_video is not None:
            # Mask out missing streams before softmax using -1e4 (safe for FP16/Half precision)
            mask_a = has_audio.float().unsqueeze(-1)
            mask_v = has_video.float().unsqueeze(-1)
            mask = torch.cat([mask_a, mask_v], dim=-1)  # [B, 2]
            raw_scores = raw_scores.masked_fill(mask == 0, -1e4)

        gates = F.softmax(raw_scores, dim=-1)
        return gates


class LAVDFDetector(nn.Module):
    """
    Unified Multimodal Detector supporting:
    - 'audio'   : Acoustic only
    - 'video'   : Visual only (ResNet18 + TSA)
    - 'fusion'  : Standard AV Concatenation
    - 'rmsav'   : Reliability-Aware Selective Fusion with TSA & CM-FCI
    """
    def __init__(
        self,
        mode: str = "rmsav",
        dim: int = 256,
        pretrained_visual: bool = False,
        use_tsa: bool = True,
        use_fci: bool = True,
    ):
        super().__init__()
        self.mode = mode.lower()
        self.dim = dim
        self.use_fci = use_fci

        if self.mode not in ["audio", "video", "fusion", "rmsav"]:
            raise ValueError(f"Unsupported mode: {mode}. Must be 'audio', 'video', 'fusion', or 'rmsav'.")

        # Visual encoder
        if self.mode in ["video", "fusion", "rmsav"]:
            self.visual_enc = VisualEncoder(dim=dim, pretrained=pretrained_visual, use_tsa=use_tsa)
            self.video_head = nn.Sequential(
                nn.Linear(dim, dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(dim // 2, 1),
            )
        else:
            self.visual_enc = None
            self.video_head = None

        # Audio encoder
        if self.mode in ["audio", "fusion", "rmsav"]:
            self.audio_enc = AudioEncoder(in_channels=1, dim=dim)
            self.audio_head = nn.Sequential(
                nn.Linear(dim, dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(dim // 2, 1),
            )
        else:
            self.audio_enc = None
            self.audio_head = None

        # Fusion & RMS-AV components
        if self.mode == "fusion":
            self.joint_head = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(dim, 1),
            )
        elif self.mode == "rmsav":
            self.reliability_gate = ReliabilityGate(dim=dim)
            if self.use_fci:
                self.fci_projector = CrossModalIncoherenceProjector(dim=dim, proj_dim=128)
            else:
                self.fci_projector = None

            self.joint_head = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(dim, 1),
            )

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        video = batch.get("video")
        audio = batch.get("audio")
        has_video = batch.get("has_video")
        has_audio = batch.get("has_audio")

        b_size = video.shape[0] if video is not None else audio.shape[0]
        device = video.device if video is not None else audio.device

        out: Dict[str, torch.Tensor] = {}

        # 1. Audio-only mode
        if self.mode == "audio":
            if audio is None:
                raise ValueError("Audio tensor is missing for 'audio' mode inference.")
            feat_a = self.audio_enc(audio)
            logit_a = self.audio_head(feat_a).squeeze(-1)
            out["logit_av"] = logit_a
            out["logit_a"] = logit_a
            return out

        # 2. Video-only mode
        if self.mode == "video":
            if video is None:
                raise ValueError("Video tensor is missing for 'video' mode inference.")
            feat_v, alpha_v = self.visual_enc(video)
            logit_v = self.video_head(feat_v).squeeze(-1)
            out["logit_av"] = logit_v
            out["logit_v"] = logit_v
            out["temporal_weights"] = alpha_v
            return out

        # 3. Multimodal modes (fusion / rmsav)
        if video is not None:
            feat_v, alpha_v = self.visual_enc(video)
        else:
            feat_v = torch.zeros(b_size, self.dim, device=device)
            alpha_v = None

        if audio is not None:
            feat_a = self.audio_enc(audio)
        else:
            feat_a = torch.zeros(b_size, self.dim, device=device)

        # Unimodal auxiliary predictions
        logit_v = self.video_head(feat_v).squeeze(-1)
        logit_a = self.audio_head(feat_a).squeeze(-1)
        out["logit_v"] = logit_v
        out["logit_a"] = logit_a
        out["temporal_weights"] = alpha_v

        if self.mode == "fusion":
            fused = torch.cat([feat_v, feat_a], dim=-1)
            logit_av = self.joint_head(fused).squeeze(-1)
            out["logit_av"] = logit_av
            out["gate"] = None
            out["incoherence"] = None
            return out

        if self.mode == "rmsav":
            # Compute Cross-Modal Incoherence if both streams present
            if self.use_fci and self.fci_projector is not None and video is not None and audio is not None:
                z_v, z_a, incoherence = self.fci_projector(feat_v, feat_a)
                out["z_v"] = z_v
                out["z_a"] = z_a
                out["incoherence"] = incoherence  # [B, 1]
            else:
                incoherence = torch.zeros(b_size, 1, device=device)
                out["z_v"] = None
                out["z_a"] = None
                out["incoherence"] = incoherence

            # Dynamic Reliability Gating
            gates = self.reliability_gate(
                feat_a, feat_v, incoherence=incoherence, has_audio=has_audio, has_video=has_video
            )
            g_a = gates[:, 0:1]
            g_v = gates[:, 1:2]

            # Selective weighting
            feat_a_gated = feat_a * g_a
            feat_v_gated = feat_v * g_v

            fused = torch.cat([feat_v_gated, feat_a_gated], dim=-1)
            logit_av = self.joint_head(fused).squeeze(-1)

            out["logit_av"] = logit_av
            out["gate"] = gates  # [B, 2] -> [g_a, g_v]
            return out

        return out


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Returns (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def build_lavdf_model(
    mode: str = "rmsav",
    dim: int = 256,
    pretrained: bool = False,
    use_tsa: bool = True,
    use_fci: bool = True,
) -> LAVDFDetector:
    """Helper factory function to instantiate model."""
    return LAVDFDetector(mode=mode, dim=dim, pretrained_visual=pretrained, use_tsa=use_tsa, use_fci=use_fci)
