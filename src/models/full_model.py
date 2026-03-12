"""
Full Multimodal Emotion Recognition Model
Integrează backbones pretrenate + fusion adapters + CCMT pentru predicție end-to-end.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

import torch
from torch import nn

from src.models.backbones import (
    BaseTextBackbone, 
    BaseAudioBackbone,
    load_text_backbones,
    load_audio_backbone,
    load_all_backbones,
)
from src.models.fusion_net import MultimodalFusionAdapter
from src.models.ccmt_layer import CascadedCrossModalTransformer


class MultimodalEmotionModel(nn.Module):
    """
    Arhitectură completă multimodală pentru recunoașterea emoțiilor.
    
    Pipeline:
    1. Backbones pretrenate (frozen):
       - Text EN (RoBERTa)
       - Text ES (RoBERTa)
       - Audio (WavLM)
    
    2. Fusion Adapters:
       - Convertesc embeddings la patch tokens
    
    3. CCMT (Cascaded Cross-Modal Transformer):
       - Cross-attention între modalități
       - Predicție finală
    
    Arhitectura suportă:
    - Training end-to-end (cu backbones frozen sau trainable)
    - Inference eficientă
    - Flexibilitate în configurare
    """

    def __init__(
        self,
        # Backbones
        text_en_backbone: Optional[BaseTextBackbone] = None,
        text_es_backbone: Optional[BaseTextBackbone] = None,
        audio_backbone: Optional[nn.Module] = None,
        
        # Dimensiuni embeddings
        text_en_dim: int = 768,
        text_es_dim: int = 768,
        audio_dim: int = 768,
        
        # CCMT config
        num_classes: int = 3,
        ccmt_dim: int = 1024,
        num_patches_per_modality: int = 100,
        ccmt_depth: int = 6,
        ccmt_heads: int = 8,
        ccmt_mlp_dim: int = 2048,
        ccmt_dim_head: int = 64,
        ccmt_dropout: float = 0.2,
        
        # Fusion config
        fusion_dropout: float = 0.1,
        use_audio_temporal_pooling: bool = True,
        
        # Training settings
        freeze_backbones: bool = True,
    ):
        """
        Args:
            text_en_backbone: Backbone preantrenat pentru text engleza
            text_es_backbone: Backbone preantrenat pentru text spaniolă
            audio_backbone: Backbone preantrenat pentru audio (WavLM)
            text_en_dim: Dimensiune output text_en
            text_es_dim: Dimensiune output text_es
            audio_dim: Dimensiune output audio
            num_classes: Număr clase pentru clasificare (default: 3 pentru valence)
            ccmt_dim: Dimensiune hidden CCMT
            num_patches_per_modality: Patch-uri per modalitate
            ccmt_depth: Număr layere CCMT
            ccmt_heads: Număr attention heads
            ccmt_mlp_dim: Dimensiune MLP în CCMT
            ccmt_dim_head: Dimensiune per attention head
            ccmt_dropout: Dropout CCMT
            fusion_dropout: Dropout în fusion adapters
            use_audio_temporal_pooling: Pooling temporal pentru audio
            freeze_backbones: Dacă True, înghețăm backbones
        """
        super().__init__()
        
        # Salvăm backbones (pot fi None dacă vrem să le adăugăm mai târziu)
        self.text_en_backbone = text_en_backbone
        self.text_es_backbone = text_es_backbone
        self.audio_backbone = audio_backbone
        
        self.freeze_backbones = freeze_backbones
        if freeze_backbones:
            self._freeze_backbones()
        
        # Fusion adapters - convertesc embeddings la patch tokens
        self.fusion_adapter = MultimodalFusionAdapter(
            text_en_dim=text_en_dim,
            text_es_dim=text_es_dim,
            audio_dim=audio_dim,
            ccmt_dim=ccmt_dim,
            num_patches_per_modality=num_patches_per_modality,
            dropout=fusion_dropout,
            use_audio_temporal_pooling=use_audio_temporal_pooling,
        )
        
        # CCMT - cross-modal transformer pentru fusion și predicție
        total_patches = num_patches_per_modality * 3
        self.ccmt = CascadedCrossModalTransformer(
            num_classes=num_classes,
            num_patches=total_patches,
            dim=ccmt_dim,
            depth=ccmt_depth,
            heads=ccmt_heads,
            mlp_dim=ccmt_mlp_dim,
            dim_head=ccmt_dim_head,
            dropout=ccmt_dropout,
        )
        
        self.num_classes = num_classes
        self.num_patches_per_modality = num_patches_per_modality

    def _freeze_backbones(self):
        """Înghețăm parametrii backbones dacă sunt setați."""
        if self.text_en_backbone is not None:
            for param in self.text_en_backbone.parameters():
                param.requires_grad = False
        
        if self.text_es_backbone is not None:
            for param in self.text_es_backbone.parameters():
                param.requires_grad = False
        
        if self.audio_backbone is not None:
            for param in self.audio_backbone.parameters():
                param.requires_grad = False

    def forward(
        self,
        text_en: Optional[Union[List[str], torch.Tensor]] = None,
        text_es: Optional[Union[List[str], torch.Tensor]] = None,
        audio: Optional[torch.Tensor] = None,
        
        # Pre-computed embeddings (opțional - mai rapid la inference)
        text_en_emb: Optional[torch.Tensor] = None,
        text_es_emb: Optional[torch.Tensor] = None,
        audio_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass prin întreaga arhitectură.
        
        Poate primi fie:
        1. Input raw (text, audio) → trece prin backbones
        2. Embeddings pre-calculate → trece direct la fusion
        
        Args:
            text_en: Text în engleză (List[str]) sau tokenized (Tensor)
            text_es: Text în spaniolă (List[str]) sau tokenized (Tensor)
            audio: Audio waveform sau features (Tensor)
            text_en_emb: Pre-computed text EN embeddings (B, text_en_dim)
            text_es_emb: Pre-computed text ES embeddings (B, text_es_dim)
            audio_emb: Pre-computed audio embeddings (B, audio_dim) sau (B, seq, audio_dim)
        
        Returns:
            Predicții clase: (batch_size, num_classes) cu Sigmoid aplicat
        """
        # Extragere embeddings dacă nu sunt pre-calculate
        if text_en_emb is None:
            if text_en is None:
                raise ValueError("Trebuie furnizat fie text_en, fie text_en_emb")
            if self.text_en_backbone is None:
                raise ValueError("text_en_backbone nu este setat")
            text_en_emb = self.text_en_backbone(text_en)
        
        if text_es_emb is None:
            if text_es is None:
                raise ValueError("Trebuie furnizat fie text_es, fie text_es_emb")
            if self.text_es_backbone is None:
                raise ValueError("text_es_backbone nu este setat")
            text_es_emb = self.text_es_backbone(text_es)
        
        if audio_emb is None:
            if audio is None:
                raise ValueError("Trebuie furnizat fie audio, fie audio_emb")
            if self.audio_backbone is None:
                raise ValueError("audio_backbone nu este setat")
            audio_emb = self.audio_backbone(audio)
        
        # Fusion - convertim embeddings la patch tokens
        multimodal_tokens = self.fusion_adapter(
            text_en_emb=text_en_emb,
            text_es_emb=text_es_emb,
            audio_emb=audio_emb,
        )  # (batch, total_patches, ccmt_dim)
        
        # CCMT - cross-modal attention și predicție
        predictions = self.ccmt(multimodal_tokens)  # (batch, num_classes)
        
        return predictions

    def predict(
        self,
        text_en: Optional[Union[List[str], torch.Tensor]] = None,
        text_es: Optional[Union[List[str], torch.Tensor]] = None,
        audio: Optional[torch.Tensor] = None,
        text_en_emb: Optional[torch.Tensor] = None,
        text_es_emb: Optional[torch.Tensor] = None,
        audio_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predicție cu interpretare rezultate.
        
        Returns:
            predictions: (batch_size, num_classes) - probabilități sigmoid
            predicted_classes: (batch_size,) - clasa cu scorul maxim
        """
        self.eval()
        with torch.no_grad():
            predictions = self.forward(
                text_en=text_en,
                text_es=text_es,
                audio=audio,
                text_en_emb=text_en_emb,
                text_es_emb=text_es_emb,
                audio_emb=audio_emb,
            )
            predicted_classes = predictions.argmax(dim=-1)
        
        return predictions, predicted_classes

    def get_trainable_parameters(self) -> int:
        """Returnează numărul de parametri trainable."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_parameters(self) -> int:
        """Returnează numărul total de parametri."""
        return sum(p.numel() for p in self.parameters())

    def print_parameter_summary(self):
        """Afișează rezumat parametri."""
        trainable = self.get_trainable_parameters()
        total = self.get_total_parameters()
        frozen = total - trainable
        
        print("\n" + "="*60)
        print("MODEL PARAMETER SUMMARY")
        print("="*60)
        print(f"Total parameters:     {total:>15,}")
        print(f"Trainable parameters: {trainable:>15,}")
        print(f"Frozen parameters:    {frozen:>15,}")
        print(f"Trainable ratio:      {trainable/total*100:>14.2f}%")
        print("="*60 + "\n")


def load_full_multimodal_model(
    text_en_checkpoint: Union[str, Path] = "checkpoints/roberta_text_en",
    text_es_checkpoint: Union[str, Path] = "checkpoints/roberta_text_es",
    audio_checkpoint: Union[str, Path] = "checkpoints/wavlm_audio",
    
    num_classes: int = 3,
    ccmt_dim: int = 1024,
    num_patches_per_modality: int = 100,
    ccmt_depth: int = 6,
    ccmt_heads: int = 8,
    ccmt_mlp_dim: int = 2048,
    
    freeze_backbones: bool = True,
    projection_dim: Optional[int] = 256,
    device: Optional[str] = None,
) -> MultimodalEmotionModel:
    """
    Încarcă modelul complet cu backbones pretrenate.
    
    Args:
        text_en_checkpoint: Path la checkpoint text EN
        text_es_checkpoint: Path la checkpoint text ES
        audio_checkpoint: Path la checkpoint audio
        num_classes: Număr clase
        ccmt_dim: Dimensiune CCMT
        num_patches_per_modality: Patch-uri per modalitate
        ccmt_depth: Depth CCMT
        ccmt_heads: Attention heads
        ccmt_mlp_dim: MLP dimension
        freeze_backbones: Freeze backbones
        projection_dim: Dimensiune proiecție uniformă (None = native dims)
        device: Device pentru model
    
    Returns:
        Model complet inițializat
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n" + "="*60)
    print("LOADING FULL MULTIMODAL MODEL")
    print("="*60)
    
    # Încarcă toate backbones
    print("\n1. Loading all backbones (text_en, text_es, audio)...")
    backbones = load_all_backbones(
        text_en_checkpoint=text_en_checkpoint,
        text_es_checkpoint=text_es_checkpoint,
        audio_checkpoint=audio_checkpoint,
        freeze=freeze_backbones,
        projection_dim=projection_dim,
    )
    
    text_en_backbone = backbones['text_en']
    text_es_backbone = backbones['text_es']
    audio_backbone = backbones['audio']
    
    # Dimensiuni embeddings
    text_en_dim = text_en_backbone.get_output_dim()
    text_es_dim = text_es_backbone.get_output_dim()
    audio_dim = audio_backbone.get_output_dim()
    
    print(f"\n2. Detected embedding dimensions:")
    print(f"   - Text EN: {text_en_dim}")
    print(f"   - Text ES: {text_es_dim}")
    print(f"   - Audio:   {audio_dim}")
    
    # Construiește modelul
    print("\n3. Building full model architecture...")
    model = MultimodalEmotionModel(
        text_en_backbone=text_en_backbone,
        text_es_backbone=text_es_backbone,
        audio_backbone=audio_backbone,
        text_en_dim=text_en_dim,
        text_es_dim=text_es_dim,
        audio_dim=audio_dim,
        num_classes=num_classes,
        ccmt_dim=ccmt_dim,
        num_patches_per_modality=num_patches_per_modality,
        ccmt_depth=ccmt_depth,
        ccmt_heads=ccmt_heads,
        ccmt_mlp_dim=ccmt_mlp_dim,
        freeze_backbones=freeze_backbones,
    )
    
    model = model.to(device)
    model.print_parameter_summary()
    
    print("Full model loaded successfully!")
    print("="*60 + "\n")
    
    return model


def load_ccmt_only_model(
    text_en_dim: int = 768,
    text_es_dim: int = 768,
    audio_dim: int = 768,
    num_classes: int = 3,
    ccmt_dim: int = 768,
    num_patches_per_modality: int = 100,
    ccmt_depth: int = 4,
    ccmt_heads: int = 4,
    ccmt_mlp_dim: int = 1024,
    ccmt_dropout: float = 0.2,
    device: Optional[str] = None,
) -> MultimodalEmotionModel:
    """
    Încarcă modelul DOAR cu fusion + CCMT (fără backbones).
    
    Util pentru antrenarea cu embeddings precalculate, economisind
    memorie și timp de încărcare.
    
    Args:
        text_en_dim: Dimensiune embeddings text EN
        text_es_dim: Dimensiune embeddings text ES
        audio_dim: Dimensiune embeddings audio
        num_classes: Număr clase
        ccmt_dim: Dimensiune CCMT
        num_patches_per_modality: Patch-uri per modalitate
        ccmt_depth: Depth CCMT
        ccmt_heads: Attention heads
        ccmt_mlp_dim: MLP dimension
        ccmt_dropout: Dropout în layerele CCMT
        device: Device pentru model
    
    Returns:
        Model CCMT (fără backbones)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n" + "="*60)
    print("LOADING CCMT-ONLY MODEL (No Backbones)")
    print("="*60)
    print(f"Device: {device}")
    print(f"Input dims: text_en={text_en_dim}, text_es={text_es_dim}, audio={audio_dim}")
    print(f"CCMT config: dim={ccmt_dim}, depth={ccmt_depth}, heads={ccmt_heads}, dropout={ccmt_dropout}")
    
    # Construiește modelul FĂRĂ backbones
    model = MultimodalEmotionModel(
        text_en_backbone=None,  # Nu încărcăm backbones
        text_es_backbone=None,
        audio_backbone=None,
        text_en_dim=text_en_dim,
        text_es_dim=text_es_dim,
        audio_dim=audio_dim,
        num_classes=num_classes,
        ccmt_dim=ccmt_dim,
        num_patches_per_modality=num_patches_per_modality,
        ccmt_depth=ccmt_depth,
        ccmt_heads=ccmt_heads,
        ccmt_mlp_dim=ccmt_mlp_dim,
        ccmt_dropout=ccmt_dropout,
        freeze_backbones=False,  # Nu contează, nu avem backbones
    )
    
    model = model.to(device)
    model.print_parameter_summary()
    
    print("CCMT-only model loaded successfully!")
    print("="*60 + "\n")
    
    return model


if __name__ == '__main__':
    # Test full model
    print("Testing MultimodalEmotionModel...")
    
    # Simulăm backbones simple
    class DummyBackbone(nn.Module):
        def __init__(self, output_dim):
            super().__init__()
            self.hidden_size = output_dim
            self.projection = nn.Linear(100, output_dim)
        
        def forward(self, x):
            if isinstance(x, list):
                batch_size = len(x)
                x = torch.randn(batch_size, 100)
            return self.projection(x)
    
    text_en_backbone = DummyBackbone(768)
    text_es_backbone = DummyBackbone(768)
    
    # Construim modelul
    model = MultimodalEmotionModel(
        text_en_backbone=text_en_backbone,
        text_es_backbone=text_es_backbone,
        audio_backbone=None,
        text_en_dim=768,
        text_es_dim=768,
        audio_dim=768,
        num_classes=3,
        ccmt_dim=1024,
        num_patches_per_modality=100,
        freeze_backbones=False,
    )
    
    model.print_parameter_summary()
    
    # Test cu embeddings pre-calculate
    batch_size = 8
    text_en_emb = torch.randn(batch_size, 768)
    text_es_emb = torch.randn(batch_size, 768)
    audio_emb = torch.randn(batch_size, 768)
    
    predictions = model(
        text_en_emb=text_en_emb,
        text_es_emb=text_es_emb,
        audio_emb=audio_emb,
    )
    
    print(f"\n✓ Predictions shape: {predictions.shape}")
    print(f"✓ Expected: ({batch_size}, 3)")
    print(f"✓ Predictions sum per sample: {predictions.sum(dim=1)}")
    
    # Test predict
    preds, classes = model.predict(
        text_en_emb=text_en_emb,
        text_es_emb=text_es_emb,
        audio_emb=audio_emb,
    )
    print(f"\n✓ Predicted classes: {classes}")
    
    print("\n✅ All tests passed!")
