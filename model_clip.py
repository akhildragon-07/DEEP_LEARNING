r"""
==================================================================================================
Multimodal Deepfake Detection: RMS-AV and Baseline Architectures
--------------------------------------------------------------------------------------------------
This module provides a unified, leak-safe, and modular implementation of:
1. RMS-AV (Reliability- and Manipulation-Aware Selective Audio-Visual Fusion)
2. VideoOnlyModel (Baseline)
3. AudioOnlyModel (Baseline)
4. StandardAVFusionModel (Vanilla Fusion Baseline)

Design Principles:
- Compatible forward interfaces across all 4 architectures.
- Explicit support for missing modalities via validity masks (no arbitrary zero-input latents).
- LayerNorm everywhere (ensuring batch-size 1 stability without BatchNorm crashes).
- Unimodal authenticity heads for fine-grained modality-level supervision.
- Reliability estimation combining validity masks with learned feature confidence.
- Compact, frozen foundation backbones (CLIP ViT-Base and Whisper-Tiny) suited for Colab T4.
==================================================================================================
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import CLIPVisionModel, WhisperModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# =========================================================================
# 1. Modular Feature Encoders (Frozen Backbones + Learnable Projectors)
# =========================================================================

class VisualEncoder(nn.Module):
    """
    Extracts high-level visual representations from video frame sequences.
    
    Backbone: Pre-trained CLIP ViT (openai/clip-vit-base-patch32), frozen.
    Temporal Aggregation: Mean pooling across sampled video frames.
    Projection: LayerNorm -> Linear -> GELU -> Dropout -> Linear into projection_dim.
    """
    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        projection_dim: int = 256,
        freeze_backbone: bool = True,
        load_backbone: bool = True,
        dropout: float = 0.1,
    ):
        super(VisualEncoder, self).__init__()
        self.projection_dim = projection_dim
        self.raw_dim = 768
        self.backbone = None

        if TRANSFORMERS_AVAILABLE and load_backbone:
            try:
                self.backbone = CLIPVisionModel.from_pretrained(clip_model_name, local_files_only=True)
                self.raw_dim = self.backbone.config.hidden_size  # 768
            except Exception:
                try:
                    self.backbone = CLIPVisionModel.from_pretrained(clip_model_name)
                    self.raw_dim = self.backbone.config.hidden_size
                except Exception:
                    self.backbone = None

            if self.backbone is not None and freeze_backbone:
                for p in self.backbone.parameters():
                    p.requires_grad = False

        self.projector = nn.Sequential(
            nn.LayerNorm(self.raw_dim),
            nn.Linear(self.raw_dim, projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(projection_dim),
        )

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        precomputed_features: Optional[torch.Tensor] = None,
        video_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, T, C, H, W) or (B * T, C, H, W) video frame tensor.
            precomputed_features: (B, raw_dim) optional pre-extracted CLIP features.
            video_mask: (B, 1) binary mask (1.0 = present, 0.0 = missing).

        Returns:
            torch.Tensor: (B, projection_dim) projected visual embedding.
        """
        if precomputed_features is not None:
            feat = precomputed_features
            b = feat.size(0)
        elif pixel_values is not None:
            b = pixel_values.size(0)
            if self.backbone is None:
                feat = torch.zeros((b, self.raw_dim), device=pixel_values.device)
            else:
                if pixel_values.dim() == 5:
                    b, t, c, h, w = pixel_values.shape
                    flat_pixels = pixel_values.view(b * t, c, h, w)
                else:
                    t = 1
                    flat_pixels = pixel_values

                with torch.no_grad():
                    outputs = self.backbone(pixel_values=flat_pixels)
                    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                        raw_tokens = outputs.pooler_output
                    else:
                        raw_tokens = outputs.last_hidden_state[:, 0, :]
                
                if pixel_values.dim() == 5:
                    raw_tokens = raw_tokens.view(b, t, -1)
                    feat = raw_tokens.mean(dim=1)
                else:
                    feat = raw_tokens
        else:
            raise ValueError("Either pixel_values or precomputed_features must be provided.")

        proj = self.projector(feat)

        # Apply validity mask if provided
        if video_mask is not None:
            proj = proj * video_mask.view(b, 1)

        return proj


class AudioEncoder(nn.Module):
    """
    Extracts phonemic and acoustic representations from log-mel spectrograms.
    
    Backbone: Pre-trained Whisper encoder (openai/whisper-tiny), frozen.
    Temporal Aggregation: Mean pooling along temporal frames.
    Projection: LayerNorm -> Linear -> GELU -> Dropout -> Linear into projection_dim.
    """
    def __init__(
        self,
        whisper_model_name: str = "openai/whisper-tiny",
        projection_dim: int = 256,
        freeze_backbone: bool = True,
        load_backbone: bool = True,
        dropout: float = 0.1,
    ):
        super(AudioEncoder, self).__init__()
        self.projection_dim = projection_dim
        self.raw_dim = 384
        self.backbone = None

        if TRANSFORMERS_AVAILABLE and load_backbone:
            try:
                whisper = WhisperModel.from_pretrained(whisper_model_name, local_files_only=True)
                self.backbone = whisper.encoder
                self.raw_dim = self.backbone.config.d_model  # 384
            except Exception:
                try:
                    whisper = WhisperModel.from_pretrained(whisper_model_name)
                    self.backbone = whisper.encoder
                    self.raw_dim = self.backbone.config.d_model
                except Exception:
                    self.backbone = None

            if self.backbone is not None and freeze_backbone:
                for p in self.backbone.parameters():
                    p.requires_grad = False
        else:
            self.backbone = None
            self.raw_dim = 384

        self.projector = nn.Sequential(
            nn.LayerNorm(self.raw_dim),
            nn.Linear(self.raw_dim, projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(projection_dim),
        )

    def forward(
        self,
        input_features: Optional[torch.Tensor] = None,
        precomputed_features: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_features: (B, 80, 3000) audio log-mel spectrogram.
            precomputed_features: (B, raw_dim) optional pre-extracted Whisper features.
            audio_mask: (B, 1) binary mask (1.0 = present, 0.0 = missing).

        Returns:
            torch.Tensor: (B, projection_dim) projected audio embedding.
        """
        if precomputed_features is not None:
            feat = precomputed_features
            b = feat.size(0)
        elif input_features is not None:
            b = input_features.size(0)
            if self.backbone is None:
                feat = torch.zeros((b, self.raw_dim), device=input_features.device)
            else:
                with torch.no_grad():
                    enc_out = self.backbone(input_features)
                    # enc_out.last_hidden_state: (B, T, d_model)
                    feat = enc_out.last_hidden_state.mean(dim=1)
        else:
            raise ValueError("Either input_features or precomputed_features must be provided.")

        proj = self.projector(feat)

        # Apply validity mask if provided
        if audio_mask is not None:
            proj = proj * audio_mask.view(b, 1)

        return proj


# =========================================================================
# 2. Prediction Heads (LayerNorm-based for Batch Size 1 Robustness)
# =========================================================================

class AuthenticityClassificationHead(nn.Module):
    """
    Multi-layer Perceptron mapping feature embeddings to scalar authenticity logits.
    Uses LayerNorm instead of BatchNorm1d to ensure stability for batch size 1.
    """
    def __init__(self, in_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super(AuthenticityClassificationHead, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits of shape (B, 1)."""
        return self.classifier(x)


# =========================================================================
# 3. RMS-AV: Reliability Estimation & Reliability-Aware Gated Fusion
# =========================================================================

class ModalityReliabilityEstimator(nn.Module):
    r"""
    Estimates the empirical forensic reliability weight r_m \in [0, 1] for a modality.
    
    Formula:
        r_m = mask_m * sigmoid(MLP_rel(z_m))
    
    If a modality is missing (mask_m = 0), reliability is strictly zero.
    If a modality is present, the MLP assesses feature consistency and informative quality.
    """
    def __init__(self, in_dim: int, hidden_dim: int = 64):
        super(ModalityReliabilityEstimator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            z: (B, in_dim) modality embedding.
            mask: (B, 1) binary mask (1.0 = valid, 0.0 = missing).

        Returns:
            torch.Tensor: (B, 1) reliability weight in [0, 1].
        """
        raw_rel = self.net(z)
        if mask is not None:
            # Strictly zero-out reliability for missing modalities
            return raw_rel * mask.view(-1, 1)
        return raw_rel


class ReliabilityAwareSelectiveFusion(nn.Module):
    """
    Fuses visual and acoustic streams guided by their estimated reliability.
    
    Mechanism:
    1. Cross-modality contextual interaction produces candidate gating vectors g_v, g_a.
    2. Reliability weights r_v, r_a dynamically scale gating contributions:
           g_v_eff = g_v * (r_v / (r_v + r_a + eps))
           g_a_eff = g_a * (r_a / (r_v + r_a + eps))
    3. Missing modalities (reliability = 0) are completely attenuated (gate = 0).
    4. If only one modality is present, it receives 100% of the active gating weight.
    """
    def __init__(self, embed_dim: int = 256, dropout: float = 0.1, eps: float = 1e-6):
        super(ReliabilityAwareSelectiveFusion, self).__init__()
        self.embed_dim = embed_dim
        self.eps = eps

        # Cross-modal gate network
        self.gate_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.Sigmoid(),
        )

        # Output projection
        self.fusion_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        z_v: torch.Tensor,
        z_a: torch.Tensor,
        r_v: torch.Tensor,
        r_a: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            z_v: (B, embed_dim) visual embedding.
            z_a: (B, embed_dim) audio embedding.
            r_v: (B, 1) visual reliability weight in [0, 1].
            r_a: (B, 1) audio reliability weight in [0, 1].

        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                - fused_rep: (B, embed_dim) fused representation.
                - (g_v_eff, g_a_eff): Effective modality gate tensors of shape (B, embed_dim).
        """
        b = z_v.size(0)
        z_joint = torch.cat([z_v, z_a], dim=-1)  # (B, 2 * embed_dim)
        raw_gates = self.gate_mlp(z_joint)        # (B, 2 * embed_dim)
        g_v, g_a = torch.split(raw_gates, self.embed_dim, dim=-1)

        # Relative reliability normalization: normalized balance
        rel_sum = r_v + r_a + self.eps
        w_v = r_v / rel_sum  # (B, 1)
        w_a = r_a / rel_sum  # (B, 1)

        # Modulate feature gates by relative reliability
        g_v_eff = g_v * w_v
        g_a_eff = g_a * w_a

        # Gated aggregation
        z_v_gated = g_v_eff * z_v
        z_a_gated = g_a_eff * z_a

        fused = self.fusion_proj(torch.cat([z_v_gated, z_a_gated], dim=-1))
        return fused, (g_v_eff, g_a_eff)


# =========================================================================
# 4. Proposed Model: RMS-AV
# =========================================================================

class RMSAV(nn.Module):
    """
    RMS-AV: Reliability- and Manipulation-Aware Selective Audio-Visual Fusion.
    
    Components:
    - Visual and Audio Encoders (frozen backbones + learnable projectors).
    - Unimodal Authenticity Heads for video and audio (auxiliary supervision).
    - Modality Reliability Estimators (derive forensic trust scores r_v, r_a).
    - Reliability-Aware Selective Gated Fusion (prevents corrupt/missing modality dominance).
    - Final Deepfake Classification Head.
    """
    def __init__(
        self,
        embed_dim: int = 256,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        whisper_model_name: str = "openai/whisper-tiny",
        dropout: float = 0.1,
        freeze_backbones: bool = True,
        load_backbones: bool = True,
    ):
        super(RMSAV, self).__init__()
        self.embed_dim = embed_dim

        # 1. Encoders
        self.visual_encoder = VisualEncoder(
            clip_model_name=clip_model_name,
            projection_dim=embed_dim,
            freeze_backbone=freeze_backbones,
            load_backbone=load_backbones,
            dropout=dropout,
        )
        self.audio_encoder = AudioEncoder(
            whisper_model_name=whisper_model_name,
            projection_dim=embed_dim,
            freeze_backbone=freeze_backbones,
            load_backbone=load_backbones,
            dropout=dropout,
        )

        # 2. Unimodal Authenticity Heads
        self.video_head = AuthenticityClassificationHead(in_dim=embed_dim, dropout=dropout)
        self.audio_head = AuthenticityClassificationHead(in_dim=embed_dim, dropout=dropout)

        # 3. Modality Reliability Estimators
        self.video_reliability_estimator = ModalityReliabilityEstimator(in_dim=embed_dim)
        self.audio_reliability_estimator = ModalityReliabilityEstimator(in_dim=embed_dim)

        # 4. Reliability-Aware Fusion
        self.fusion = ReliabilityAwareSelectiveFusion(embed_dim=embed_dim, dropout=dropout)

        # 5. Final Deepfake Classification Head
        self.classifier = AuthenticityClassificationHead(in_dim=embed_dim, dropout=dropout)

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        input_features: Optional[torch.Tensor] = None,
        precomputed_visual: Optional[torch.Tensor] = None,
        precomputed_audio: Optional[torch.Tensor] = None,
        video_mask: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Unified forward pass supporting missing modalities, precomputed features, and raw inputs.
        """
        # Determine batch size and device
        for t in (pixel_values, input_features, precomputed_visual, precomputed_audio):
            if t is not None:
                b = t.size(0)
                device = t.device
                break
        else:
            raise ValueError("At least one input tensor must be provided.")

        # Default masks to 1.0 if tensor is provided, 0.0 if None
        if video_mask is None:
            v_val = 1.0 if (pixel_values is not None or precomputed_visual is not None) else 0.0
            video_mask = torch.full((b, 1), v_val, device=device)
        else:
            video_mask = video_mask.view(b, 1).to(device)

        if audio_mask is None:
            a_val = 1.0 if (input_features is not None or precomputed_audio is not None) else 0.0
            audio_mask = torch.full((b, 1), a_val, device=device)
        else:
            audio_mask = audio_mask.view(b, 1).to(device)

        # Visual feature extraction
        if pixel_values is not None or precomputed_visual is not None:
            z_v = self.visual_encoder(pixel_values, precomputed_visual, video_mask)
        else:
            z_v = torch.zeros((b, self.embed_dim), device=device)

        # Audio feature extraction
        if input_features is not None or precomputed_audio is not None:
            z_a = self.audio_encoder(input_features, precomputed_audio, audio_mask)
        else:
            z_a = torch.zeros((b, self.embed_dim), device=device)

        # Unimodal authenticity predictions
        video_logits = self.video_head(z_v)
        audio_logits = self.audio_head(z_a)

        # Reliability estimation (strictly 0.0 when mask is 0.0)
        r_v = self.video_reliability_estimator(z_v, video_mask)
        r_a = self.audio_reliability_estimator(z_a, audio_mask)

        # Reliability-aware gated fusion
        z_fused, (g_v, g_a) = self.fusion(z_v, z_a, r_v, r_a)

        # Final classification
        cls_logits = self.classifier(z_fused)
        cls_probs = torch.sigmoid(cls_logits)

        return {
            "cls_logits": cls_logits,
            "cls_probs": cls_probs,
            "video_logits": video_logits,
            "audio_logits": audio_logits,
            "reliability_video": r_v,
            "reliability_audio": r_a,
            "gate_weights": (g_v, g_a),
            "fused_features": z_fused,
        }


# =========================================================================
# 5. Baseline Models: Video-Only, Audio-Only, Standard AV Fusion
# =========================================================================

class VideoOnlyModel(nn.Module):
    """
    Video-Only Baseline: Utilizes identical visual encoder and classification head.
    """
    def __init__(
        self,
        embed_dim: int = 256,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        dropout: float = 0.1,
        freeze_backbones: bool = True,
        load_backbones: bool = True,
    ):
        super(VideoOnlyModel, self).__init__()
        self.embed_dim = embed_dim
        self.visual_encoder = VisualEncoder(
            clip_model_name=clip_model_name,
            projection_dim=embed_dim,
            freeze_backbone=freeze_backbones,
            load_backbone=load_backbones,
            dropout=dropout,
        )
        self.classifier = AuthenticityClassificationHead(in_dim=embed_dim, dropout=dropout)

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        input_features: Optional[torch.Tensor] = None,
        precomputed_visual: Optional[torch.Tensor] = None,
        precomputed_audio: Optional[torch.Tensor] = None,
        video_mask: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Union[torch.Tensor, None]]:
        for t in (pixel_values, precomputed_visual):
            if t is not None:
                b = t.size(0)
                device = t.device
                break
        else:
            raise ValueError("Video-only model requires pixel_values or precomputed_visual.")

        if video_mask is None:
            video_mask = torch.ones((b, 1), device=device)
        else:
            video_mask = video_mask.view(b, 1).to(device)

        z_v = self.visual_encoder(pixel_values, precomputed_visual, video_mask)
        logits = self.classifier(z_v)
        probs = torch.sigmoid(logits)

        return {
            "cls_logits": logits,
            "cls_probs": probs,
            "video_logits": logits,
            "audio_logits": None,
            "reliability_video": video_mask,
            "reliability_audio": None,
            "fused_features": z_v,
        }


class AudioOnlyModel(nn.Module):
    """
    Audio-Only Baseline: Utilizes identical acoustic encoder and classification head.
    """
    def __init__(
        self,
        embed_dim: int = 256,
        whisper_model_name: str = "openai/whisper-tiny",
        dropout: float = 0.1,
        freeze_backbones: bool = True,
        load_backbones: bool = True,
    ):
        super(AudioOnlyModel, self).__init__()
        self.embed_dim = embed_dim
        self.audio_encoder = AudioEncoder(
            whisper_model_name=whisper_model_name,
            projection_dim=embed_dim,
            freeze_backbone=freeze_backbones,
            load_backbone=load_backbones,
            dropout=dropout,
        )
        self.classifier = AuthenticityClassificationHead(in_dim=embed_dim, dropout=dropout)

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        input_features: Optional[torch.Tensor] = None,
        precomputed_visual: Optional[torch.Tensor] = None,
        precomputed_audio: Optional[torch.Tensor] = None,
        video_mask: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Union[torch.Tensor, None]]:
        for t in (input_features, precomputed_audio):
            if t is not None:
                b = t.size(0)
                device = t.device
                break
        else:
            raise ValueError("Audio-only model requires input_features or precomputed_audio.")

        if audio_mask is None:
            audio_mask = torch.ones((b, 1), device=device)
        else:
            audio_mask = audio_mask.view(b, 1).to(device)

        z_a = self.audio_encoder(input_features, precomputed_audio, audio_mask)
        logits = self.classifier(z_a)
        probs = torch.sigmoid(logits)

        return {
            "cls_logits": logits,
            "cls_probs": probs,
            "video_logits": None,
            "audio_logits": logits,
            "reliability_video": None,
            "reliability_audio": audio_mask,
            "fused_features": z_a,
        }


class StandardAVFusionModel(nn.Module):
    """
    Standard Audio-Visual Fusion Baseline:
    Direct concatenation of visual and acoustic projected embeddings followed by LayerNorm + MLP.
    No reliability estimation, no dynamic gating suppression.
    """
    def __init__(
        self,
        embed_dim: int = 256,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        whisper_model_name: str = "openai/whisper-tiny",
        dropout: float = 0.1,
        freeze_backbones: bool = True,
        load_backbones: bool = True,
    ):
        super(StandardAVFusionModel, self).__init__()
        self.embed_dim = embed_dim

        self.visual_encoder = VisualEncoder(
            clip_model_name=clip_model_name,
            projection_dim=embed_dim,
            freeze_backbone=freeze_backbones,
            load_backbone=load_backbones,
            dropout=dropout,
        )
        self.audio_encoder = AudioEncoder(
            whisper_model_name=whisper_model_name,
            projection_dim=embed_dim,
            freeze_backbone=freeze_backbones,
            load_backbone=load_backbones,
            dropout=dropout,
        )

        # Standard linear concatenation fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = AuthenticityClassificationHead(in_dim=embed_dim, dropout=dropout)

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        input_features: Optional[torch.Tensor] = None,
        precomputed_visual: Optional[torch.Tensor] = None,
        precomputed_audio: Optional[torch.Tensor] = None,
        video_mask: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Union[torch.Tensor, None]]:
        for t in (pixel_values, input_features, precomputed_visual, precomputed_audio):
            if t is not None:
                b = t.size(0)
                device = t.device
                break
        else:
            raise ValueError("At least one input tensor must be provided.")

        if video_mask is None:
            v_val = 1.0 if (pixel_values is not None or precomputed_visual is not None) else 0.0
            video_mask = torch.full((b, 1), v_val, device=device)
        else:
            video_mask = video_mask.view(b, 1).to(device)

        if audio_mask is None:
            a_val = 1.0 if (input_features is not None or precomputed_audio is not None) else 0.0
            audio_mask = torch.full((b, 1), a_val, device=device)
        else:
            audio_mask = audio_mask.view(b, 1).to(device)

        if pixel_values is not None or precomputed_visual is not None:
            z_v = self.visual_encoder(pixel_values, precomputed_visual, video_mask)
        else:
            z_v = torch.zeros((b, self.embed_dim), device=device)

        if input_features is not None or precomputed_audio is not None:
            z_a = self.audio_encoder(input_features, precomputed_audio, audio_mask)
        else:
            z_a = torch.zeros((b, self.embed_dim), device=device)

        z_cat = torch.cat([z_v, z_a], dim=-1)
        z_fused = self.fusion_layer(z_cat)

        logits = self.classifier(z_fused)
        probs = torch.sigmoid(logits)

        return {
            "cls_logits": logits,
            "cls_probs": probs,
            "video_logits": None,
            "audio_logits": None,
            "reliability_video": video_mask,
            "reliability_audio": audio_mask,
            "fused_features": z_fused,
        }


# =========================================================================
# 6. Factory Builder Function
# =========================================================================

def build_model(
    model_type: str = "rms_av",
    embed_dim: int = 256,
    clip_model_name: str = "openai/clip-vit-base-patch32",
    whisper_model_name: str = "openai/whisper-tiny",
    dropout: float = 0.1,
    freeze_backbones: bool = True,
    load_backbones: bool = True,
) -> nn.Module:
    """
    Model factory function for RMS-AV and Baselines.
    
    Args:
        model_type: One of ['rms_av', 'video_only', 'audio_only', 'standard_fusion'].
        embed_dim: Common projection dimension.
        clip_model_name: Hugging Face visual model name.
        whisper_model_name: Hugging Face acoustic model name.
        dropout: Regularization dropout probability.
        freeze_backbones: Whether backbones are frozen.
        load_backbones: Whether to load heavy foundation backbones (set False for precomputed tests).
    """
    m_type = model_type.lower().strip()
    if m_type == "rms_av":
        return RMSAV(
            embed_dim=embed_dim,
            clip_model_name=clip_model_name,
            whisper_model_name=whisper_model_name,
            dropout=dropout,
            freeze_backbones=freeze_backbones,
            load_backbones=load_backbones,
        )
    elif m_type == "video_only":
        return VideoOnlyModel(
            embed_dim=embed_dim,
            clip_model_name=clip_model_name,
            dropout=dropout,
            freeze_backbones=freeze_backbones,
            load_backbones=load_backbones,
        )
    elif m_type == "audio_only":
        return AudioOnlyModel(
            embed_dim=embed_dim,
            whisper_model_name=whisper_model_name,
            dropout=dropout,
            freeze_backbones=freeze_backbones,
            load_backbones=load_backbones,
        )
    elif m_type in ["standard_fusion", "standard_av", "vanilla_fusion"]:
        return StandardAVFusionModel(
            embed_dim=embed_dim,
            clip_model_name=clip_model_name,
            whisper_model_name=whisper_model_name,
            dropout=dropout,
            freeze_backbones=freeze_backbones,
            load_backbones=load_backbones,
        )
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Choose from: 'rms_av', 'video_only', 'audio_only', 'standard_fusion'."
        )
