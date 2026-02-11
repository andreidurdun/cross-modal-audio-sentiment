"""Multi-modal CCMT network that fuses audio and text channels."""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import BertTokenizer, CamembertTokenizer

from .backbones import BERTBackbone, CamemBERTBackbone, WavLMBackbone
from .ccmt_layer import CCMTLayer
from src.utils.token_sampling import sample_tokens


class CCMTModel(nn.Module):
    """Complete Cross-modal Cross-attention Transformer model for emotion recognition."""

    def __init__(
        self,
        audio_backbone: str = "microsoft/wavlm-base-plus",
        text_en_backbone: str = "bert-base-uncased",
        text_fr_backbone: str = "camembert-base",
        num_classes: int = 4,
        hidden_dim: int = 768,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        token_sample_k: int = 100,
        freeze_backbones: bool = False,
    ) -> None:
        super().__init__()
        self.token_sample_k = token_sample_k

        self.audio_encoder = WavLMBackbone(audio_backbone, freeze=freeze_backbones)
        self.text_en_encoder = BERTBackbone(text_en_backbone, freeze=freeze_backbones)
        self.text_fr_encoder = CamemBERTBackbone(text_fr_backbone, freeze=freeze_backbones)

        self.audio_proj = nn.Linear(self.audio_encoder.hidden_size, hidden_dim)
        self.text_en_proj = nn.Linear(self.text_en_encoder.hidden_size, hidden_dim)
        self.text_fr_proj = nn.Linear(self.text_fr_encoder.hidden_size, hidden_dim)

        self.ccmt_layer = CCMTLayer(hidden_dim, num_attention_heads, dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self.text_en_tokenizer = BertTokenizer.from_pretrained(text_en_backbone)
        self.text_fr_tokenizer = CamembertTokenizer.from_pretrained(text_fr_backbone)

    def forward(self, audio, text_en, text_fr):
        batch_size = audio.size(0)

        audio_features = self.audio_encoder(audio)
        audio_features = self.audio_proj(audio_features)

        text_en_inputs = self.text_en_tokenizer(
            text_en,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        ).to(audio.device)

        text_en_features = self.text_en_encoder(
            text_en_inputs['input_ids'],
            text_en_inputs['attention_mask'],
        )
        text_en_features = self.text_en_proj(text_en_features)

        text_fr_inputs = self.text_fr_tokenizer(
            text_fr,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        ).to(audio.device)

        text_fr_features = self.text_fr_encoder(
            text_fr_inputs['input_ids'],
            text_fr_inputs['attention_mask'],
        )
        text_fr_features = self.text_fr_proj(text_fr_features)

        audio_sampled = sample_tokens(audio_features, self.token_sample_k)
        text_en_sampled = sample_tokens(text_en_features, self.token_sample_k)
        text_fr_sampled = sample_tokens(text_fr_features, self.token_sample_k)

        audio_fused, text_en_fused, text_fr_fused = self.ccmt_layer(
            audio_sampled,
            text_en_sampled,
            text_fr_sampled,
            text_en_mask=None,
            text_fr_mask=None,
        )

        audio_pooled = audio_fused.mean(dim=1)
        text_en_pooled = text_en_fused.mean(dim=1)
        text_fr_pooled = text_fr_fused.mean(dim=1)

        fused = torch.cat([audio_pooled, text_en_pooled, text_fr_pooled], dim=-1)
        logits = self.classifier(fused)
        return logits
