"""
Fusion Network - Adaptoare pentru integrarea features multimodale în CCMT.
Convertirea embeddings din backbones în token patches pentru CCMT.
"""
from __future__ import annotations

import torch
from torch import nn
from typing import Dict, List, Optional


class ModalityAdapter(nn.Module):
    """
    Adaptor pentru transformarea embeddings într-un număr fix de patch-uri (tokens).
    Folosește proiecție + reshape pentru a genera reprezentări patch-based.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1024,
        num_patches: int = 100,
        dropout: float = 0.1,
    ):
        """
        Args:
            input_dim: Dimensiunea embedding-ului de intrare (e.g., 768, 256)
            output_dim: Dimensiunea de ieșire pe token/patch (trebuie să coincidă cu CCMT dim)
            num_patches: Număr de patch-uri/tokens generate pentru modalitate
            dropout: Dropout rate pentru training
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_patches = num_patches

        # Proiectăm embedding-ul la numărul necesar de patch-uri × dimensiunea dorită
        intermediate_dim = num_patches * output_dim

        self.projection = nn.Sequential(
            nn.Linear(input_dim, intermediate_dim),
            nn.LayerNorm(intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Embeddings de la backbone de tip (batch_size, input_dim)
        
        Returns:
            Patch tokens de tip (batch_size, num_patches, output_dim)
        """
        batch_size = x.size(0)
        
        # Proiectăm la spațiul multipatches
        x = self.projection(x)  # (B, num_patches * output_dim)
        
        # Reshape pentru patch structure
        x = x.view(batch_size, self.num_patches, self.output_dim)  # (B, num_patches, dim)
        
        return x


class AudioAdapter(nn.Module):
    """
    Adaptor specializat pentru audio - poate prelucra embeddings secvențiale sau pooled.
    Suportă reducerea temporală pentru WavLM features.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1024,
        num_patches: int = 100,
        dropout: float = 0.1,
        use_temporal_pooling: bool = False,
    ):
        """
        Args:
            input_dim: Dimensiunea features audio (e.g., 768 pentru WavLM)
            output_dim: Dimensiunea de ieșire per patch
            num_patches: Număr patch-uri generate
            dropout: Dropout rate
            use_temporal_pooling: Dacă True, aplică pooling temporal pe secvențe lungi
        """
        super().__init__()
        self.use_temporal_pooling = use_temporal_pooling
        self.num_patches = num_patches
        self.output_dim = output_dim
        
        if use_temporal_pooling:
            # Pentru features secvențiale, aggregăm temporal apoi generăm patches
            self.temporal_pool = nn.AdaptiveAvgPool1d(1)
            
        # Proiect audio la patch space
        intermediate_dim = num_patches * output_dim
        self.projection = nn.Sequential(
            nn.Linear(input_dim, intermediate_dim),
            nn.LayerNorm(intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Audio features - poate fi:
               - (batch_size, hidden_dim) - pooled
               - (batch_size, seq_len, hidden_dim) - secvențial
        
        Returns:
            Audio patch tokens (batch_size, num_patches, output_dim)
        """
        # Dacă primim features secvențiale și avem pooling activat
        if x.dim() == 3 and self.use_temporal_pooling:
            # (B, seq, dim) → (B, dim, seq)
            x = x.transpose(1, 2)
            # Pool temporal → (B, dim, 1)
            x = self.temporal_pool(x)
            # Flatten → (B, dim)
            x = x.squeeze(-1)
        elif x.dim() == 3:
            # Dacă nu vrem pooling, facem mean pooling simplu
            x = x.mean(dim=1)  # (B, dim)
        
        batch_size = x.size(0)
        
        # Proiectăm la patch space
        x = self.projection(x)  # (B, num_patches * output_dim)
        
        # Reshape la patch structure
        x = x.view(batch_size, self.num_patches, self.output_dim)
        
        return x


class MultimodalFusionAdapter(nn.Module):
    """
    Adaptor complet pentru fuziunea multimodală - 3 modalități:
    - Text EN (RoBERTa-en)
    - Text ES (RoBERTa-es)  
    - Audio (WavLM)
    
    Convertește toate embeddings la un număr fix de patches pentru CCMT.
    """

    def __init__(
        self,
        modality_dims: Dict[str, int],
        modalities: List[str],
        ccmt_dim: int = 1024,
        num_patches_per_modality: int = 100,
        dropout: float = 0.1,
        use_audio_temporal_pooling: bool = False,
    ):
        """
        Args:
            modality_dims: Dimensiuni embeddings pentru fiecare modalitate activa
            modalities: Ordinea modalitatilor folosita la concatenarea tokenilor
            ccmt_dim: Dimensiune token CCMT (trebuie consistentă)
            num_patches_per_modality: Patch-uri per modalitate
            dropout: Dropout rate
            use_audio_temporal_pooling: Pooling temporal pentru audio
        """
        super().__init__()
        if not modalities:
            raise ValueError("MultimodalFusionAdapter necesita cel putin o modalitate")

        missing_dims = [modality for modality in modalities if modality not in modality_dims]
        if missing_dims:
            raise ValueError(f"Lipsesc dimensiuni pentru modalitati: {missing_dims}")

        self.modalities = list(modalities)
        self.adapters = nn.ModuleDict()
        for modality in self.modalities:
            if modality == "audio":
                self.adapters[modality] = AudioAdapter(
                    input_dim=modality_dims[modality],
                    output_dim=ccmt_dim,
                    num_patches=num_patches_per_modality,
                    dropout=dropout,
                    use_temporal_pooling=use_audio_temporal_pooling,
                )
            else:
                self.adapters[modality] = ModalityAdapter(
                    input_dim=modality_dims[modality],
                    output_dim=ccmt_dim,
                    num_patches=num_patches_per_modality,
                    dropout=dropout,
                )

        self.num_patches_per_modality = num_patches_per_modality
        self.total_patches = num_patches_per_modality * len(self.modalities)

    def forward(
        self,
        modality_embeddings: Optional[Dict[str, torch.Tensor]] = None,
        **legacy_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            modality_embeddings: Mapping intre modalitate si embedding
        
        Returns:
            Multimodal token patches: (batch_size, total_patches, ccmt_dim)
            Ordinea urmeaza lista `modalities`
        """
        modality_embeddings = dict(modality_embeddings or {})
        for key, value in legacy_embeddings.items():
            if key.endswith("_emb"):
                modality_embeddings[key[:-4]] = value

        missing_modalities = [
            modality for modality in self.modalities if modality not in modality_embeddings
        ]
        if missing_modalities:
            raise ValueError(f"Lipsesc embeddings pentru modalitati: {missing_modalities}")

        modality_patches = [
            self.adapters[modality](modality_embeddings[modality])
            for modality in self.modalities
        ]
        return torch.cat(modality_patches, dim=1)

    def get_total_patches(self) -> int:
        """Returnează numărul total de patch-uri/tokens generate."""
        return self.total_patches


if __name__ == '__main__':
    # Test fusion adapter
    print("Testing MultimodalFusionAdapter...")
    
    batch_size = 16
    ccmt_dim = 1024
    num_patches = 100
    modalities = ["text_en", "text_es", "audio"]
    modality_dims = {
        "text_en": 768,
        "text_es": 768,
        "audio": 768,
    }
    
    adapter = MultimodalFusionAdapter(
        modality_dims=modality_dims,
        modalities=modalities,
        ccmt_dim=ccmt_dim,
        num_patches_per_modality=num_patches,
    )
    
    # Simulăm embeddings
    text_en_emb = torch.randn(batch_size, modality_dims["text_en"])
    text_es_emb = torch.randn(batch_size, modality_dims["text_es"])
    audio_emb = torch.randn(batch_size, modality_dims["audio"])
    
    # Forward pass
    multimodal_tokens = adapter(
        text_en_emb=text_en_emb,
        text_es_emb=text_es_emb,
        audio_emb=audio_emb,
    )
    
    print(f"✓ Input shapes:")
    print(f"  text_en: {text_en_emb.shape}")
    print(f"  text_es: {text_es_emb.shape}")
    print(f"  audio: {audio_emb.shape}")
    print(f"✓ Output shape: {multimodal_tokens.shape}")
    print(f"✓ Expected: (batch={batch_size}, patches={num_patches * len(modalities)}, dim={ccmt_dim})")
    
    assert multimodal_tokens.shape == (batch_size, num_patches * len(modalities), ccmt_dim)
    print("\n✅ All tests passed!")
