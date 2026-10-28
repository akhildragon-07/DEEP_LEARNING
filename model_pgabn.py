r"""
==================================================================================================
Prompt-Guided Asymmetric Bridge Network (PG-ABN) for Multimodal Deepfake Detection
--------------------------------------------------------------------------------------------------
Architecture Specification & IEEE-Style Documentation:
Author: Multimodal Deepfake Detection Research Team
Target Venue: IEEE Transactions on Information Forensics and Security (T-IFS) / IEEE Access

Theoretical Framework:
1. Frozen Vision-Language and Audio Foundation Backbones:
   - Visual Stream: CLIP (openai/clip-vit-base-patch32) frozen to retain generic semantic features.
   - Acoustic Stream: Whisper (openai/whisper-tiny) encoder frozen to capture phonemic-acoustic patterns.
   - Backbone freezing (requires_grad=False) prevents catastrophic forgetting and drastically reduces
     training footprint while maintaining robust generalization.

2. Antigravity Prompts (P_v, P_a):
   - Continuous learnable parameter tensors injected into frozen multimodal representations via
     residual connections:
         \tilde{F}_v = F_v + \lambda_v \cdot P_v
         \tilde{F}_a = F_a + \lambda_a \cdot P_a
   - These prompts guide frozen latent distributions toward forensic manipulation boundaries
     without modifying backbone weights.

3. Asymmetric Gated Fusion Network (AGFN):
   - Aligns heterogeneous dimensionalities into a shared latent metric space \mathbb{R}^{D_{bridge}}.
   - Addresses the "Negative Transfer" pathology in unilateral deepfakes (e.g., synthesized face with
     authentic audio, or cloned voice with genuine video). Symmetric aggregation causes unmanipulated,
     clean modalities to overpower subtle forgery indicators.
   - Evaluates cross-modal correlation and generates asymmetric gating coefficients [g_v, g_a] \in [0, 1]:
         [g_v, g_a] = \sigma(W_g [\phi_v(\tilde{F}_v) \,||\, \phi_a(\tilde{F}_a)] + b_g)
         \hat{H}_v = g_v \odot \phi_v(\tilde{F}_v)
         \hat{H}_a = g_a \odot \phi_a(\tilde{F}_a)
   - Clean modalities are dynamically attenuated, isolating forgery artifacts in the corrupted stream.

4. Multi-Task Forensic Heads:
   - Binary Classification Head: Predicts probability of manipulation \hat{y}_{cls} \in [0, 1].
   - Temporal Localization Head: Predicts boundary coordinates [\tau_{start}, \tau_{end}] \in [0, 1]^2
     via Sigmoid activation for temporal deepfake grounding.
==================================================================================================
"""

import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel, WhisperModel


class AntigravityPrompt(nn.Module):
    r"""
    IEEE-Style Specification:
    Class: AntigravityPrompt
    Description:
        Implements learnable continuous prompt injection into high-dimensional representations
        extracted from frozen foundation encoders. Prompts act as forensic condition vectors
        steered via backpropagation, bridging pre-trained foundation distributions with
        forgery artifact manifolds via a parameterized residual connection.

    Mathematical Formulation:
        \tilde{F} = F + \alpha \odot P
        where F \in \mathbb{R}^{B \times D} represents backbone features,
              P \in \mathbb{R}^{1 \times D} denotes the trainable Antigravity Prompt,
              \alpha \in \mathbb{R}^{1} is an optional learnable scaling coefficient.

    Parameters:
        dim (int): Feature dimensionality matching backbone output embedding space.
        learnable_scale (bool): Whether to include an adaptive scaling parameter. Default: True.
        init_std (float): Standard deviation for normal initialization of prompt vector.
    """

    def __init__(self, dim: int, learnable_scale: bool = True, init_std: float = 0.02):
        super(AntigravityPrompt, self).__init__()
        self.dim = dim
        self.prompt = nn.Parameter(torch.empty(1, dim))
        nn.init.trunc_normal_(self.prompt, std=init_std)

        if learnable_scale:
            self.scale = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
        else:
            self.register_buffer("scale", torch.tensor([1.0], dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Injects the Antigravity Prompt into the input feature tensor via residual addition.

        Args:
            x (torch.Tensor): Feature tensor of shape (B, D) or (B, T, D).

        Returns:
            torch.Tensor: Prompt-conditioned feature tensor with identical shape to x.
        """
        return x + self.scale * self.prompt


class AsymmetricGatedFusion(nn.Module):
    r"""
    IEEE-Style Specification:
    Class: AsymmetricGatedFusion
    Description:
        Asymmetric Gated Fusion Network designed to mitigate the "Negative Transfer" problem
        inherent to unilateral deepfakes (where only audio or video is faked while the counterpart
        stream remains pristine). Standard symmetric fusion functions allow dominant clean signals
        to mask weak forgery cues. This network aligns cross-modal feature dimensions, analyzes
        joint context, and derives bounded asymmetric gating matrices to selectively suppress
        unmanipulated streams while prioritizing artifact-laden modalities.

    Mathematical Formulation:
        H_v = \text{LayerNorm}(\text{Linear}_v(\tilde{F}_v)) \in \mathbb{R}^{B \times D_{bridge}}
        H_a = \text{LayerNorm}(\text{Linear}_a(\tilde{F}_a)) \in \mathbb{R}^{B \times D_{bridge}}
        H_{joint} = [H_v \,||\, H_a] \in \mathbb{R}^{B \times 2 D_{bridge}}
        G = \sigma(\text{MLP}_{gate}(H_{joint})) \in [0, 1]^{B \times 2 D_{bridge}}
        [g_v, g_a] = \text{Split}(G, D_{bridge})
        \hat{H}_v = g_v \odot H_v
        \hat{H}_a = g_a \odot H_a
        F_{fused} = [\hat{H}_v \,||\, \hat{H}_a] \in \mathbb{R}^{B \times 2 D_{bridge}}

    Parameters:
        visual_dim (int): Input dimensionality of prompt-guided visual features.
        audio_dim (int): Input dimensionality of prompt-guided audio features.
        bridge_dim (int): Common dimension for cross-modal projection. Default: 512.
        dropout (float): Dropout probability for regularization. Default: 0.1.
    """

    def __init__(
        self,
        visual_dim: int = 768,
        audio_dim: int = 384,
        bridge_dim: int = 512,
        dropout: float = 0.1,
    ):
        super(AsymmetricGatedFusion, self).__init__()
        self.bridge_dim = bridge_dim

        # Modality alignment projectors with normalization
        self.visual_align = nn.Sequential(
            nn.Linear(visual_dim, bridge_dim),
            nn.LayerNorm(bridge_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.audio_align = nn.Sequential(
            nn.Linear(audio_dim, bridge_dim),
            nn.LayerNorm(bridge_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Asymmetric Gating Network to identify and suppress clean modalities
        self.gate_network = nn.Sequential(
            nn.Linear(bridge_dim * 2, bridge_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bridge_dim, bridge_dim * 2),
            nn.Sigmoid(),
        )

        # Cross-modal projection refinement
        self.fusion_norm = nn.LayerNorm(bridge_dim * 2)

    def forward(
        self, visual_feat: torch.Tensor, audio_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Executes asymmetric cross-modal alignment and gated suppression.

        Args:
            visual_feat (torch.Tensor): Visual representations of shape (B, visual_dim).
            audio_feat (torch.Tensor): Acoustic representations of shape (B, audio_dim).

        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                - Fused representation tensor of shape (B, 2 * bridge_dim).
                - Tuple of gating tensors (g_v, g_a), each of shape (B, bridge_dim).
        """
        # Step 1: Metric alignment
        h_v = self.visual_align(visual_feat)  # (B, bridge_dim)
        h_a = self.audio_align(audio_feat)   # (B, bridge_dim)

        # Step 2: Cross-modal concatenation
        h_joint = torch.cat([h_v, h_a], dim=-1)  # (B, 2 * bridge_dim)

        # Step 3: Compute asymmetric suppression gates
        gates = self.gate_network(h_joint)  # (B, 2 * bridge_dim) \in [0, 1]
        g_v, g_a = torch.split(gates, self.bridge_dim, dim=-1)

        # Step 4: Modulate modalities to filter unmanipulated noise
        h_v_gated = g_v * h_v
        h_a_gated = g_a * h_a

        # Step 5: Fused representation
        fused = self.fusion_norm(torch.cat([h_v_gated, h_a_gated], dim=-1))
        return fused, (g_v, g_a)


class BinaryClassificationHead(nn.Module):
    r"""
    IEEE-Style Specification:
    Class: BinaryClassificationHead
    Description:
        Multi-Layer Perceptron (MLP) mapping fused multimodal embeddings to a single
        scalar logit representing the probability of manipulation:
            P(\text{Fake} \mid V, A) = \sigma(\text{logit})

    Parameters:
        in_dim (int): Dimensionality of incoming fused representations.
        hidden_dim (int): Dimensionality of internal hidden representation. Default: 256.
        dropout (float): Dropout probability. Default: 0.1.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super(BinaryClassificationHead, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing raw classification logits.

        Args:
            x (torch.Tensor): Fused feature tensor of shape (B, in_dim).

        Returns:
            torch.Tensor: Unnormalized scalar logits of shape (B, 1).
        """
        return self.classifier(x)


class TemporalLocalizationHead(nn.Module):
    r"""
    IEEE-Style Specification:
    Class: TemporalLocalizationHead
    Description:
        Forensic boundary regressor estimating normalized temporal onset and offset
        timestamps [\tau_{start}, \tau_{end}] \in [0, 1]^2 corresponding to the deepfake
        manipulation interval within the multimodal clip. Sigmoid activation ensures strictly
        bounded coordinates within video bounds.

    Parameters:
        in_dim (int): Dimensionality of incoming fused representations.
        hidden_dim (int): Dimensionality of internal hidden projection. Default: 128.
        dropout (float): Dropout probability. Default: 0.1.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super(TemporalLocalizationHead, self).__init__()
        self.locator = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
            nn.Sigmoid(),  # Strict bounding to [0, 1] normalized temporal range
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predicts normalized manipulation intervals.

        Args:
            x (torch.Tensor): Fused feature tensor of shape (B, in_dim).

        Returns:
            torch.Tensor: Normalized coordinates [start_time, end_time] of shape (B, 2).
        """
        return self.locator(x)


class PGABN(nn.Module):
    r"""
    IEEE-Style Specification:
    Class: PGABN (Prompt-Guided Asymmetric Bridge Network)
    Description:
        Complete architecture combining frozen multimodal foundational backbones,
        trainable Antigravity Prompts, an Asymmetric Gated Fusion Network to prevent
        negative transfer from pristine streams, and dual forensic heads for binary forgery
        classification and temporal manipulation localization.

    Structural Pipeline:
        1. Input Video (B, T=16, C=3, H=224, W=224) -> Frozen CLIP-ViT -> Mean Temporal Pool -> (B, 768).
        2. Input Audio (B, 80, 3000) -> Frozen Whisper-tiny Encoder -> Mean Pooling -> (B, 384).
        3. Antigravity Prompt Injection:
               \tilde{F}_v = F_v + P_v
               \tilde{F}_a = F_a + P_a
        4. Asymmetric Gated Fusion -> Suppress Clean Modality -> (B, 2 * bridge_dim).
        5. Dual Prediction Heads:
               - Binary Logits \in \mathbb{R}^{B \times 1}
               - Temporal Boundaries [\tau_{start}, \tau_{end}] \in [0, 1]^{B \times 2}

    Parameters:
        clip_model_name (str): Hugging Face identifier for vision backbone.
                               Default: "openai/clip-vit-base-patch32".
        whisper_model_name (str): Hugging Face identifier for acoustic backbone.
                                  Default: "openai/whisper-tiny".
        bridge_dim (int): Shared dimensional bridge size. Default: 512.
        dropout (float): Dropout probability throughout fusion and classification heads.
        freeze_backbones (bool): Whether to freeze backbone weights. Default: True.
    """

    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        whisper_model_name: str = "openai/whisper-tiny",
        bridge_dim: int = 512,
        dropout: float = 0.1,
        freeze_backbones: bool = True,
    ):
        super(PGABN, self).__init__()

        # =========================================================================
        # 1. Foundation Backbones (Visual & Audio)
        # =========================================================================
        # Visual Stream: CLIP ViT-Base-Patch32
        self.vision_encoder = CLIPVisionModel.from_pretrained(clip_model_name)
        self.visual_dim = self.vision_encoder.config.hidden_size  # 768

        # Acoustic Stream: Whisper-Tiny Encoder
        whisper = WhisperModel.from_pretrained(whisper_model_name)
        self.audio_encoder = whisper.encoder
        self.audio_dim = self.audio_encoder.config.d_model  # 384

        # Freeze backbones as required
        if freeze_backbones:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            for param in self.audio_encoder.parameters():
                param.requires_grad = False

        # =========================================================================
        # 2. Antigravity Prompts (Trainable residual conditioning tensors)
        # =========================================================================
        self.visual_prompt = AntigravityPrompt(dim=self.visual_dim)
        self.audio_prompt = AntigravityPrompt(dim=self.audio_dim)

        # =========================================================================
        # 3. Asymmetric Gated Fusion Network
        # =========================================================================
        self.asymmetric_fusion = AsymmetricGatedFusion(
            visual_dim=self.visual_dim,
            audio_dim=self.audio_dim,
            bridge_dim=bridge_dim,
            dropout=dropout,
        )

        # =========================================================================
        # 4. Multi-Task Forensic Prediction Heads
        # =========================================================================
        fused_dim = bridge_dim * 2
        self.classifier_head = BinaryClassificationHead(
            in_dim=fused_dim, hidden_dim=256, dropout=dropout
        )
        self.localization_head = TemporalLocalizationHead(
            in_dim=fused_dim, hidden_dim=128, dropout=dropout
        )

    def extract_visual_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Extracts temporal-aggregated visual features from video frame batches.

        Args:
            pixel_values (torch.Tensor): Video frames formatted as (B, T, C, H, W)
                                         or flattened as (B * T, C, H, W).

        Returns:
            torch.Tensor: Video-level visual feature vector of shape (B, 768).
        """
        if pixel_values.dim() == 5:
            b, t, c, h, w = pixel_values.shape
            pixel_values = pixel_values.view(b * t, c, h, w)
            is_video_sequence = True
        else:
            b = pixel_values.shape[0]
            is_video_sequence = False

        # Inference through frozen CLIP ViT backbone
        with torch.no_grad():
            vision_outputs = self.vision_encoder(pixel_values=pixel_values)
            # Use pooler output or CLS token embedding
            if hasattr(vision_outputs, "pooler_output") and vision_outputs.pooler_output is not None:
                frame_features = vision_outputs.pooler_output
            else:
                frame_features = vision_outputs.last_hidden_state[:, 0, :]

        if is_video_sequence:
            # Reshape back to (B, T, D) and average across temporal dimension
            frame_features = frame_features.view(b, t, -1)
            video_features = frame_features.mean(dim=1)
        else:
            video_features = frame_features

        return video_features

    def extract_audio_features(self, input_features: torch.Tensor) -> torch.Tensor:
        """
        Extracts time-aggregated acoustic representations from log-mel spectrogram features.

        Args:
            input_features (torch.Tensor): Log-mel audio spectrogram of shape (B, 80, 3000).

        Returns:
            torch.Tensor: Clip-level audio feature vector of shape (B, 384).
        """
        with torch.no_grad():
            encoder_outputs = self.audio_encoder(input_features)
            # encoder_outputs.last_hidden_state has shape (B, T_audio, 384)
            audio_seq = encoder_outputs.last_hidden_state
            # Mean pooling over temporal dimension
            audio_features = audio_seq.mean(dim=1)
        return audio_features

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_features: torch.Tensor,
    ) -> Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward propagation across the PG-ABN network.

        Args:
            pixel_values (torch.Tensor): Normalized video frames tensor (B, 16, 3, 224, 224).
            input_features (torch.Tensor): Audio log-mel spectrogram (B, 80, 3000).

        Returns:
            Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
                - 'cls_logits': Binary classification logits of shape (B, 1).
                - 'cls_probs': Sigmoid-activated probabilities of shape (B, 1).
                - 'localization': Normalized temporal interval [tau_start, tau_end] (B, 2).
                - 'gate_weights': Tuple of (visual_gate, audio_gate) tensors.
                - 'fused_features': Fused cross-modal representation (B, 2 * bridge_dim).
        """
        # Step 1: Feature extraction via frozen backbones
        f_v = self.extract_visual_features(pixel_values)  # (B, 768)
        f_a = self.extract_audio_features(input_features) # (B, 384)

        # Step 2: Inject Antigravity Prompts via residual conditioning
        f_v_prompted = self.visual_prompt(f_v)  # (B, 768)
        f_a_prompted = self.audio_prompt(f_a)   # (B, 384)

        # Step 3: Asymmetric Gated Fusion with dynamic clean-modality suppression
        fused_rep, (g_v, g_a) = self.asymmetric_fusion(f_v_prompted, f_a_prompted)

        # Step 4: Multi-task forensic prediction heads
        cls_logits = self.classifier_head(fused_rep)
        cls_probs = torch.sigmoid(cls_logits)
        loc_coords = self.localization_head(fused_rep)

        return {
            "cls_logits": cls_logits,
            "cls_probs": cls_probs,
            "localization": loc_coords,
            "gate_weights": (g_v, g_a),
            "fused_features": fused_rep,
        }


def build_pgabn_model(
    clip_model_name: str = "openai/clip-vit-base-patch32",
    whisper_model_name: str = "openai/whisper-tiny",
    bridge_dim: int = 512,
    dropout: float = 0.1,
) -> PGABN:
    """
    Factory helper function to instantiate and verify the PG-ABN architecture.

    Returns:
        PGABN: Initialized network with frozen backbones and trainable bridge prompts.
    """
    return PGABN(
        clip_model_name=clip_model_name,
        whisper_model_name=whisper_model_name,
        bridge_dim=bridge_dim,
        dropout=dropout,
        freeze_backbones=True,
    )
