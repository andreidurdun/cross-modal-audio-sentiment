"""Model components: backbones, fusion networks, adapters."""

from src.models.backbones import (
    BaseTextBackbone,
    BaseAudioBackbone,
    TextENBackbone, 
    TextESBackbone,
    TextDEBackbone,
    TextFRBackbone,
    AudioWavLMBackbone,
    load_text_en_backbone,
    load_text_es_backbone,
    load_text_de_backbone,
    load_text_fr_backbone,
    load_text_backbones,
    load_audio_backbone,
    load_all_backbones,
)

from src.models.fusion_net import (
    ModalityAdapter,
    AudioAdapter,
    MultimodalFusionAdapter,
)

from src.models.ccmt_layer import (
    CascadedCrossModalTransformer,
    Attention,
    Transformer,
    PreNorm,
    FeedForward,
)

from src.models.full_model import (
    MultimodalEmotionModel,
    load_full_multimodal_model,
    load_ccmt_only_model,
)

__all__ = [
    # Backbones
    "BaseTextBackbone",
    "BaseAudioBackbone",
    "TextENBackbone",
    "TextESBackbone",
    "TextDEBackbone",
    "TextFRBackbone",
    "AudioWavLMBackbone",
    "load_text_en_backbone",
    "load_text_es_backbone", 
    "load_text_de_backbone",
    "load_text_fr_backbone",
    "load_text_backbones",
    "load_audio_backbone",
    "load_all_backbones",
    
    # Fusion adapters
    "ModalityAdapter",
    "AudioAdapter",
    "MultimodalFusionAdapter",
    
    # CCMT
    "CascadedCrossModalTransformer",
    "Attention",
    "Transformer",
    "PreNorm",
    "FeedForward",
    
    # Full model
    "MultimodalEmotionModel",
    "load_full_multimodal_model",
    "load_ccmt_only_model",
]
