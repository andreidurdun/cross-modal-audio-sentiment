from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoModelForAudioClassification, AutoFeatureExtractor


def _infer_peft_num_labels(model_path: Path) -> Optional[int]:
    adapter_weights_path = model_path / "adapter_model.safetensors"
    if not adapter_weights_path.exists():
        return None

    try:
        from safetensors import safe_open
    except ImportError:
        return None

    candidate_suffixes = (
        "classifier.modules_to_save.default.weight",
        "score.modules_to_save.default.weight",
        "classifier.weight",
        "score.weight",
    )

    with safe_open(str(adapter_weights_path), framework="pt", device="cpu") as file_handle:
        for key in file_handle.keys():
            if key.endswith(candidate_suffixes):
                return int(file_handle.get_tensor(key).shape[0])

    return None


def _load_text_encoder_model(model_name: str) -> nn.Module:
    """
    Load a text encoder compatible with pooled embedding extraction.

    If a PEFT adapter checkpoint is detected, load it with PEFT and extract the
    underlying Roberta encoder so LoRA weights are effectively applied.
    """
    model_path = Path(model_name)
    adapter_config = model_path / "adapter_config.json"

    if adapter_config.exists():
        from peft import AutoPeftModelForSequenceClassification

        def _extract_encoder(model: nn.Module) -> nn.Module:
            base_prefix = getattr(model, "base_model_prefix", None)
            if base_prefix and hasattr(model, base_prefix):
                return getattr(model, base_prefix)

            for attribute_name in ("roberta", "bert", "camembert", "deberta", "distilbert"):
                if hasattr(model, attribute_name):
                    return getattr(model, attribute_name)

            raise ValueError(f"Unsupported PEFT base model in checkpoint: {model_name}")

        inferred_num_labels = _infer_peft_num_labels(model_path)
        peft_load_kwargs = {"ignore_mismatched_sizes": True}
        if inferred_num_labels is not None:
            peft_load_kwargs["num_labels"] = inferred_num_labels

        peft_model = AutoPeftModelForSequenceClassification.from_pretrained(
            model_name,
            **peft_load_kwargs,
        )
        if hasattr(peft_model, "merge_and_unload"):
            merged_model = peft_model.merge_and_unload()
            return _extract_encoder(merged_model)

        base_model = peft_model.base_model.model
        return _extract_encoder(base_model)

    return AutoModel.from_pretrained(model_name)


def _mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return summed / counts


class BaseTextBackbone(nn.Module):
    """
    Base text backbone optimized for multimodal ensemble learning.
    
    Features:
    - Pretrained text encoder (BERT-like)
    - Optional projection layer for unified representation space
    - Batch processing for large datasets
    - Optional freezing for feature extraction mode
    """

    def __init__(
        self,
        model_name: str,
        max_length: int = 128,
        use_cls_pooling: bool = False,
        freeze: bool = True,
        projection_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.use_cls_pooling = use_cls_pooling

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = _load_text_encoder_model(model_name)
        self.hidden_size = self.model.config.hidden_size
        
        # Optional projection layer for ensemble/fusion
        if projection_dim is not None and projection_dim != self.hidden_size:
            self.projection = nn.Linear(self.hidden_size, projection_dim)
            self.output_dim = projection_dim
        else:
            self.projection = None
            self.output_dim = self.hidden_size

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def get_output_dim(self) -> int:
        return self.output_dim

    def forward(
        self,
        inputs: Union[List[str], Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Forward pass returning pooled embeddings.
        
        Args:
            inputs: List of strings or tokenizer dict with input_ids, attention_mask
        
        Returns:
            Pooled embeddings of shape (batch_size, output_dim)
        """
        if isinstance(inputs, dict):
            encodings = inputs
        else:
            encodings = self.tokenizer(
                inputs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

        device = next(self.model.parameters()).device
        encodings = {key: value.to(device) for key, value in encodings.items()}

        outputs = self.model(**encodings, return_dict=True)

        if self.use_cls_pooling and getattr(outputs, "pooler_output", None) is not None:
            embeddings = outputs.pooler_output
        else:
            embeddings = _mean_pooling(outputs.last_hidden_state, encodings["attention_mask"])

        if self.projection is not None:
            embeddings = self.projection(embeddings)

        return embeddings
    
    def batch_forward(
        self,
        texts: List[str],
        batch_size: int = 32,
        return_raw: bool = False,
    ) -> torch.Tensor:
        """
        Process large batch of texts in chunks to avoid memory overflow.
        
        Args:
            texts: List of text strings
            batch_size: Processing batch size
            return_raw: If True, return raw embeddings before projection
        
        Returns:
            Stacked embeddings of shape (len(texts), output_dim)
        """
        all_embeddings = []
        self.eval()
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                
                encodings = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                
                device = next(self.model.parameters()).device
                encodings = {key: value.to(device) for key, value in encodings.items()}
                
                outputs = self.model(**encodings, return_dict=True)
                
                if self.use_cls_pooling and getattr(outputs, "pooler_output", None) is not None:
                    embeddings = outputs.pooler_output
                else:
                    embeddings = _mean_pooling(outputs.last_hidden_state, encodings["attention_mask"])
                
                if not return_raw and self.projection is not None:
                    embeddings = self.projection(embeddings)
                
                all_embeddings.append(embeddings.cpu())
        
        return torch.cat(all_embeddings, dim=0)


class TextENBackbone(BaseTextBackbone):
    """English text encoder backbone - optimized for sentiment analysis."""

    def __init__(
        self,
        model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        max_length: int = 128,
        use_cls_pooling: bool = False,
        freeze: bool = True,
        projection_dim: Optional[int] = 256,
    ) -> None:
        super().__init__(
            model_name=model_name,
            max_length=max_length,
            use_cls_pooling=use_cls_pooling,
            freeze=freeze,
            projection_dim=projection_dim,
        )


class TextESBackbone(BaseTextBackbone):
    """Spanish text encoder backbone - optimized for sentiment analysis."""

    def __init__(
        self,
        model_name: str = "pysentimiento/robertuito-sentiment-analysis",
        max_length: int = 128,
        use_cls_pooling: bool = False,
        freeze: bool = True,
        projection_dim: Optional[int] = 256,
    ) -> None:
        super().__init__(
            model_name=model_name,
            max_length=max_length,
            use_cls_pooling=use_cls_pooling,
            freeze=freeze,
            projection_dim=projection_dim,
        )


class TextDEBackbone(BaseTextBackbone):
    """German text encoder backbone."""

    def __init__(
        self,
        model_name: str = "deepset/gbert-base",
        max_length: int = 128,
        use_cls_pooling: bool = False,
        freeze: bool = True,
        projection_dim: Optional[int] = 256,
    ) -> None:
        super().__init__(
            model_name=model_name,
            max_length=max_length,
            use_cls_pooling=use_cls_pooling,
            freeze=freeze,
            projection_dim=projection_dim,
        )


class TextFRBackbone(BaseTextBackbone):
    """French text encoder backbone."""

    def __init__(
        self,
        model_name: str = "almanach/camembert-base",
        max_length: int = 128,
        use_cls_pooling: bool = False,
        freeze: bool = True,
        projection_dim: Optional[int] = 256,
    ) -> None:
        super().__init__(
            model_name=model_name,
            max_length=max_length,
            use_cls_pooling=use_cls_pooling,
            freeze=freeze,
            projection_dim=projection_dim,
        )


def load_text_en_backbone(
    checkpoint_dir: Union[str, Path] = "checkpoints/roberta_text_en",
    freeze: bool = True,
    projection_dim: Optional[int] = 256,
) -> TextENBackbone:
    """Load pretrained English text backbone from checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    best_model_path = checkpoint_dir / "best_model"
    
    if not best_model_path.exists():
        raise FileNotFoundError(f"Model not found at {best_model_path}")
    
    backbone = TextENBackbone(
        model_name=str(best_model_path),
        freeze=freeze,
        projection_dim=projection_dim,
    )
    print(f"Loaded English text backbone from {best_model_path}")
    print(f"  Output dim: {backbone.get_output_dim()}")
    return backbone


def load_text_es_backbone(
    checkpoint_dir: Union[str, Path] = "checkpoints/roberta_text_es",
    freeze: bool = True,
    projection_dim: Optional[int] = 256,
) -> TextESBackbone:
    """Load pretrained Spanish text backbone from checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    best_model_path = checkpoint_dir / "best_model"
    
    if not best_model_path.exists():
        raise FileNotFoundError(f"Model not found at {best_model_path}")
    
    backbone = TextESBackbone(
        model_name=str(best_model_path),
        freeze=freeze,
        projection_dim=projection_dim,
    )
    print(f"Loaded Spanish text backbone from {best_model_path}")
    print(f"  Output dim: {backbone.get_output_dim()}")
    return backbone


def load_text_de_backbone(
    checkpoint_dir: Union[str, Path] = "checkpoints/roberta_text_de",
    freeze: bool = True,
    projection_dim: Optional[int] = 256,
) -> TextDEBackbone:
    """Load pretrained German text backbone from checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    best_model_path = checkpoint_dir / "best_model"

    if not best_model_path.exists():
        raise FileNotFoundError(f"Model not found at {best_model_path}")

    backbone = TextDEBackbone(
        model_name=str(best_model_path),
        freeze=freeze,
        projection_dim=projection_dim,
    )
    print(f"Loaded German text backbone from {best_model_path}")
    print(f"  Output dim: {backbone.get_output_dim()}")
    return backbone


def load_text_fr_backbone(
    checkpoint_dir: Union[str, Path] = "checkpoints/roberta_text_fr",
    freeze: bool = True,
    projection_dim: Optional[int] = 256,
) -> TextFRBackbone:
    """Load pretrained French text backbone from checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    best_model_path = checkpoint_dir / "best_model"

    if not best_model_path.exists():
        raise FileNotFoundError(f"Model not found at {best_model_path}")

    backbone = TextFRBackbone(
        model_name=str(best_model_path),
        freeze=freeze,
        projection_dim=projection_dim,
    )
    print(f"Loaded French text backbone from {best_model_path}")
    print(f"  Output dim: {backbone.get_output_dim()}")
    return backbone


class BaseAudioBackbone(nn.Module):
    """
    Base audio backbone optimized for multimodal ensemble learning.
    
    Features:
    - Pretrained audio encoder (WavLM, Wav2Vec2, etc.)
    - Optional projection layer for unified representation space
    - Support for sequence and pooled outputs
    - Optional freezing for feature extraction mode
    """

    def __init__(
        self,
        model_name: str,
        freeze: bool = True,
        projection_dim: Optional[int] = None,
        use_pooled_output: bool = True,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.use_pooled_output = use_pooled_output

        # Load pretrained audio model
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModelForAudioClassification.from_pretrained(
            model_name,
            num_labels=3,  # placeholder, won't use the classifier
            ignore_mismatched_sizes=True,
        )
        
        # Get hidden size from the base model (not classifier)
        if hasattr(self.model, 'wavlm'):
            self.hidden_size = self.model.wavlm.config.hidden_size
        elif hasattr(self.model, 'wav2vec2'):
            self.hidden_size = self.model.wav2vec2.config.hidden_size
        elif hasattr(self.model, 'config'):
            self.hidden_size = self.model.config.hidden_size
        else:
            raise ValueError(f"Cannot determine hidden size for model {model_name}")
        
        # Optional projection layer for ensemble/fusion
        if projection_dim is not None and projection_dim != self.hidden_size:
            self.projection = nn.Linear(self.hidden_size, projection_dim)
            self.output_dim = projection_dim
        else:
            self.projection = None
            self.output_dim = self.hidden_size

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            # Keep projection trainable if it exists
            if self.projection is not None:
                for param in self.projection.parameters():
                    param.requires_grad = True

    def get_output_dim(self) -> int:
        """Get embedding output dimension."""
        return self.output_dim

    def forward(
        self,
        audio: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass returning audio embeddings.
        
        Args:
            audio: Raw audio tensors of shape (batch_size, sequence_length)
            attention_mask: Optional attention mask
        
        Returns:
            Embeddings of shape (batch_size, output_dim) if use_pooled_output
            or (batch_size, seq_len, output_dim) for sequence output
        """
        device = next(self.model.parameters()).device
        audio = audio.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # Get base model (wavlm, wav2vec2, etc.)
        if hasattr(self.model, 'wavlm'):
            base_model = self.model.wavlm
        elif hasattr(self.model, 'wav2vec2'):
            base_model = self.model.wav2vec2
        else:
            raise ValueError("Unsupported audio model architecture")

        # Forward through base model
        outputs = base_model(
            audio,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Get hidden states
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_size)

        if self.use_pooled_output:
            # Mean pooling over sequence dimension
            if attention_mask is not None:
                if attention_mask.shape[1] != hidden_states.shape[1]:
                    if hasattr(base_model, "_get_feature_vector_attention_mask"):
                        attention_mask = base_model._get_feature_vector_attention_mask(
                            hidden_states.shape[1], attention_mask
                        )
                    else:
                        attention_mask = attention_mask[:, : hidden_states.shape[1]]
                mask = attention_mask.unsqueeze(-1).type_as(hidden_states)
                summed = (hidden_states * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1)
                embeddings = summed / counts
            else:
                embeddings = hidden_states.mean(dim=1)
        else:
            embeddings = hidden_states

        if self.projection is not None:
            embeddings = self.projection(embeddings)

        return embeddings


class AudioWavLMBackbone(BaseAudioBackbone):
    """WavLM audio encoder backbone - optimized for emotion recognition."""

    def __init__(
        self,
        model_name: str = "microsoft/wavlm-base-plus",
        freeze: bool = True,
        projection_dim: Optional[int] = 256,
        use_pooled_output: bool = True,
    ) -> None:
        super().__init__(
            model_name=model_name,
            freeze=freeze,
            projection_dim=projection_dim,
            use_pooled_output=use_pooled_output,
        )


def load_audio_backbone(
    checkpoint_dir: Union[str, Path] = "checkpoints/wavlm_audio",
    freeze: bool = True,
    projection_dim: Optional[int] = 256,
    use_pooled_output: bool = True,
) -> AudioWavLMBackbone:
    """Load pretrained audio backbone from checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    best_model_path = checkpoint_dir / "best_model"
    
    if not best_model_path.exists():
        raise FileNotFoundError(f"Model not found at {best_model_path}")
    
    backbone = AudioWavLMBackbone(
        model_name=str(best_model_path),
        freeze=freeze,
        projection_dim=projection_dim,
        use_pooled_output=use_pooled_output,
    )
    print(f"Loaded audio backbone from {best_model_path}")
    print(f"  Output dim: {backbone.get_output_dim()}")
    return backbone


def load_text_backbones(
    en_checkpoint_dir: Union[str, Path] = "checkpoints/roberta_text_en",
    es_checkpoint_dir: Union[str, Path] = "checkpoints/roberta_text_es",
    freeze: bool = True,
    projection_dim: Optional[int] = 256,
) -> Dict[str, BaseTextBackbone]:
    """Load both English and Spanish text backbones for ensemble."""
    backbones = {
        'text_en': load_text_en_backbone(
            en_checkpoint_dir, 
            freeze=freeze,
            projection_dim=projection_dim,
        ),
        'text_es': load_text_es_backbone(
            es_checkpoint_dir, 
            freeze=freeze,
            projection_dim=projection_dim,
        ),
    }
    print(f"\nBoth text backbones loaded successfully!")
    print(f"  text_en output: {backbones['text_en'].get_output_dim()}")
    print(f"  text_es output: {backbones['text_es'].get_output_dim()}")
    return backbones


def load_all_backbones(
    text_en_checkpoint: Union[str, Path] = "checkpoints/roberta_text_en",
    text_es_checkpoint: Union[str, Path] = "checkpoints/roberta_text_es",
    text_de_checkpoint: Union[str, Path] = "checkpoints/roberta_text_de",
    text_fr_checkpoint: Union[str, Path] = "checkpoints/roberta_text_fr",
    audio_checkpoint: Union[str, Path] = "checkpoints/wavlm_audio",
    freeze: bool = True,
    projection_dim: Optional[int] = 256,
    modalities: Optional[List[str]] = None,
) -> Dict[str, nn.Module]:
    """Load requested backbones for multimodal fusion."""
    print("\n" + "="*60)
    print("Loading All Multimodal Backbones")
    print("="*60)
    modalities = list(modalities or ["text_en", "text_es", "audio"])
    loaders = {
        'text_en': lambda: load_text_en_backbone(
            text_en_checkpoint,
            freeze=freeze,
            projection_dim=projection_dim,
        ),
        'text_es': lambda: load_text_es_backbone(
            text_es_checkpoint,
            freeze=freeze,
            projection_dim=projection_dim,
        ),
        'text_de': lambda: load_text_de_backbone(
            text_de_checkpoint,
            freeze=freeze,
            projection_dim=projection_dim,
        ),
        'text_fr': lambda: load_text_fr_backbone(
            text_fr_checkpoint,
            freeze=freeze,
            projection_dim=projection_dim,
        ),
        'audio': lambda: load_audio_backbone(
            audio_checkpoint,
            freeze=freeze,
            projection_dim=projection_dim,
        ),
    }

    invalid_modalities = set(modalities) - set(loaders)
    if invalid_modalities:
        raise ValueError(f"Unsupported modalities requested: {sorted(invalid_modalities)}")

    backbones = {modality: loaders[modality]() for modality in modalities}
    
    print(f"\n" + "="*60)
    print(f"All requested backbones loaded successfully!")
    for modality, backbone in backbones.items():
        print(f"  {modality}: {backbone.get_output_dim()} dim")
    print("="*60 + "\n")
    
    return backbones
