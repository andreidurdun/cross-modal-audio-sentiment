"""
Cross-modal Cross-attention Transformer (CCMT) layer implementation.
"""

import torch
import torch.nn as nn
import math


class CrossAttention(nn.Module):
    """
    Cross-attention mechanism between query and key-value modalities.
    """
    
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        """
        Args:
            hidden_dim: Dimension of hidden states
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Linear projections for query, key, value
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, key_mask=None):
        """
        Forward pass for cross-attention.
        
        Args:
            query: Query tensor of shape (batch_size, query_len, hidden_dim)
            key: Key tensor of shape (batch_size, key_len, hidden_dim)
            value: Value tensor of shape (batch_size, value_len, hidden_dim)
            key_mask: Optional mask for keys (batch_size, key_len)
        
        Returns:
            Output tensor of shape (batch_size, query_len, hidden_dim)
        """
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.q_proj(query)  # (batch_size, query_len, hidden_dim)
        K = self.k_proj(key)    # (batch_size, key_len, hidden_dim)
        V = self.v_proj(value)  # (batch_size, value_len, hidden_dim)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if key_mask is not None:
            key_mask = key_mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, key_len)
            scores = scores.masked_fill(key_mask == 0, float('-inf'))
        
        # Compute attention weights
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, V)  # (batch_size, num_heads, query_len, head_dim)
        
        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        
        # Output projection
        out = self.out_proj(out)
        
        return out


class CCMTLayer(nn.Module):
    """
    Cross-modal Cross-attention Transformer layer.
    Implements attention between three modalities: audio, text_en, text_fr.
    """
    
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        """
        Args:
            hidden_dim: Dimension of hidden states
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        # Cross-attention layers
        # Audio attends to text_en
        self.audio_to_text_en = CrossAttention(hidden_dim, num_heads, dropout)
        # Audio attends to text_fr
        self.audio_to_text_fr = CrossAttention(hidden_dim, num_heads, dropout)
        # Text_en attends to audio
        self.text_en_to_audio = CrossAttention(hidden_dim, num_heads, dropout)
        # Text_fr attends to audio
        self.text_fr_to_audio = CrossAttention(hidden_dim, num_heads, dropout)
        # Text_en attends to text_fr
        self.text_en_to_text_fr = CrossAttention(hidden_dim, num_heads, dropout)
        # Text_fr attends to text_en
        self.text_fr_to_text_en = CrossAttention(hidden_dim, num_heads, dropout)
        
        # Feed-forward networks
        self.ffn_audio = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.ffn_text_en = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.ffn_text_fr = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm_audio_1 = nn.LayerNorm(hidden_dim)
        self.norm_audio_2 = nn.LayerNorm(hidden_dim)
        self.norm_text_en_1 = nn.LayerNorm(hidden_dim)
        self.norm_text_en_2 = nn.LayerNorm(hidden_dim)
        self.norm_text_fr_1 = nn.LayerNorm(hidden_dim)
        self.norm_text_fr_2 = nn.LayerNorm(hidden_dim)
    
    def forward(self, audio_features, text_en_features, text_fr_features,
                text_en_mask=None, text_fr_mask=None):
        """
        Forward pass through CCMT layer.
        
        Args:
            audio_features: Audio features (batch_size, audio_len, hidden_dim)
            text_en_features: English text features (batch_size, text_en_len, hidden_dim)
            text_fr_features: French text features (batch_size, text_fr_len, hidden_dim)
            text_en_mask: Optional mask for English text
            text_fr_mask: Optional mask for French text
        
        Returns:
            Fused features for each modality
        """
        # Audio cross-attention
        audio_from_text_en = self.audio_to_text_en(
            audio_features, text_en_features, text_en_features, text_en_mask
        )
        audio_from_text_fr = self.audio_to_text_fr(
            audio_features, text_fr_features, text_fr_features, text_fr_mask
        )
        audio_fused = audio_features + audio_from_text_en + audio_from_text_fr
        audio_fused = self.norm_audio_1(audio_fused)
        audio_fused = audio_fused + self.ffn_audio(audio_fused)
        audio_fused = self.norm_audio_2(audio_fused)
        
        # Text_en cross-attention
        text_en_from_audio = self.text_en_to_audio(
            text_en_features, audio_features, audio_features
        )
        text_en_from_text_fr = self.text_en_to_text_fr(
            text_en_features, text_fr_features, text_fr_features, text_fr_mask
        )
        text_en_fused = text_en_features + text_en_from_audio + text_en_from_text_fr
        text_en_fused = self.norm_text_en_1(text_en_fused)
        text_en_fused = text_en_fused + self.ffn_text_en(text_en_fused)
        text_en_fused = self.norm_text_en_2(text_en_fused)
        
        # Text_fr cross-attention
        text_fr_from_audio = self.text_fr_to_audio(
            text_fr_features, audio_features, audio_features
        )
        text_fr_from_text_en = self.text_fr_to_text_en(
            text_fr_features, text_en_features, text_en_features, text_en_mask
        )
        text_fr_fused = text_fr_features + text_fr_from_audio + text_fr_from_text_en
        text_fr_fused = self.norm_text_fr_1(text_fr_fused)
        text_fr_fused = text_fr_fused + self.ffn_text_fr(text_fr_fused)
        text_fr_fused = self.norm_text_fr_2(text_fr_fused)
        
        return audio_fused, text_en_fused, text_fr_fused


if __name__ == "__main__":
    # Test CCMT layer
    batch_size = 2
    audio_len = 50
    text_en_len = 30
    text_fr_len = 35
    hidden_dim = 768
    
    ccmt = CCMTLayer(hidden_dim)
    
    audio_features = torch.randn(batch_size, audio_len, hidden_dim)
    text_en_features = torch.randn(batch_size, text_en_len, hidden_dim)
    text_fr_features = torch.randn(batch_size, text_fr_len, hidden_dim)
    
    audio_fused, text_en_fused, text_fr_fused = ccmt(
        audio_features, text_en_features, text_fr_features
    )
    
    print(f"Audio fused shape: {audio_fused.shape}")
    print(f"Text EN fused shape: {text_en_fused.shape}")
    print(f"Text FR fused shape: {text_fr_fused.shape}")
