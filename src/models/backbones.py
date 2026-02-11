"""
Wrapper classes for pretrained backbone models.
Includes WavLM for audio, BERT for English text, and CamemBERT for French text.
"""

import torch
import torch.nn as nn
from transformers import (
    Wav2Vec2Model,
    BertModel,
    CamembertModel,
    AutoModel
)


class WavLMBackbone(nn.Module):
    """
    Wrapper for WavLM audio encoder.
    """
    
    def __init__(self, model_name="microsoft/wavlm-base-plus", freeze=False):
        """
        Args:
            model_name: HuggingFace model name
            freeze: Whether to freeze backbone parameters
        """
        super().__init__()
        
        self.model = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.model.config.hidden_size
        
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
    
    def forward(self, audio):
        """
        Forward pass through WavLM.
        
        Args:
            audio: Tensor of shape (batch_size, audio_length)
        
        Returns:
            Tensor of shape (batch_size, num_frames, hidden_size)
        """
        outputs = self.model(audio, output_hidden_states=True)
        
        # Use last hidden state
        hidden_states = outputs.last_hidden_state
        
        return hidden_states


class BERTBackbone(nn.Module):
    """
    Wrapper for BERT text encoder (English).
    """
    
    def __init__(self, model_name="bert-base-uncased", freeze=False):
        """
        Args:
            model_name: HuggingFace model name
            freeze: Whether to freeze backbone parameters
        """
        super().__init__()
        
        self.model = BertModel.from_pretrained(model_name)
        self.hidden_size = self.model.config.hidden_size
        
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass through BERT.
        
        Args:
            input_ids: Tensor of shape (batch_size, seq_len)
            attention_mask: Tensor of shape (batch_size, seq_len)
        
        Returns:
            Tensor of shape (batch_size, seq_len, hidden_size)
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Use last hidden state
        hidden_states = outputs.last_hidden_state
        
        return hidden_states


class CamemBERTBackbone(nn.Module):
    """
    Wrapper for CamemBERT text encoder (French).
    """
    
    def __init__(self, model_name="camembert-base", freeze=False):
        """
        Args:
            model_name: HuggingFace model name
            freeze: Whether to freeze backbone parameters
        """
        super().__init__()
        
        self.model = CamembertModel.from_pretrained(model_name)
        self.hidden_size = self.model.config.hidden_size
        
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass through CamemBERT.
        
        Args:
            input_ids: Tensor of shape (batch_size, seq_len)
            attention_mask: Tensor of shape (batch_size, seq_len)
        
        Returns:
            Tensor of shape (batch_size, seq_len, hidden_size)
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Use last hidden state
        hidden_states = outputs.last_hidden_state
        
        return hidden_states


def test_backbones():
    """Test backbone models."""
    batch_size = 2
    
    # Test WavLM
    print("Testing WavLM...")
    wavlm = WavLMBackbone()
    audio = torch.randn(batch_size, 16000)  # 1 second of audio at 16kHz
    audio_features = wavlm(audio)
    print(f"WavLM output shape: {audio_features.shape}")
    
    # Test BERT
    print("\nTesting BERT...")
    bert = BERTBackbone()
    input_ids = torch.randint(0, 30000, (batch_size, 50))
    attention_mask = torch.ones(batch_size, 50)
    text_en_features = bert(input_ids, attention_mask)
    print(f"BERT output shape: {text_en_features.shape}")
    
    # Test CamemBERT
    print("\nTesting CamemBERT...")
    camembert = CamemBERTBackbone()
    input_ids = torch.randint(0, 32000, (batch_size, 50))
    attention_mask = torch.ones(batch_size, 50)
    text_fr_features = camembert(input_ids, attention_mask)
    print(f"CamemBERT output shape: {text_fr_features.shape}")


if __name__ == "__main__":
    test_backbones()
