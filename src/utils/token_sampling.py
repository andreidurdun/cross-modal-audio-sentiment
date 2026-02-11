"""
Token sampling utilities.
Implements random selection of k tokens from sequence features.
"""

import torch


def sample_tokens(features, k, dim=1):
    """
    Randomly sample k tokens from sequence features.
    
    Args:
        features: Feature tensor of shape (batch_size, seq_len, hidden_dim)
        k: Number of tokens to sample
        dim: Dimension along which to sample (default: 1 for sequence dimension)
    
    Returns:
        Sampled features of shape (batch_size, k, hidden_dim)
    """
    batch_size, seq_len, hidden_dim = features.shape
    
    # If k >= seq_len, return all features
    if k >= seq_len:
        return features
    
    # Sample k random indices for each batch
    sampled_features = []
    for i in range(batch_size):
        # Generate random indices
        indices = torch.randperm(seq_len, device=features.device)[:k]
        indices = indices.sort()[0]  # Sort indices to maintain temporal order (optional)
        
        # Sample features
        sampled = features[i, indices, :]
        sampled_features.append(sampled)
    
    # Stack into batch
    sampled_features = torch.stack(sampled_features, dim=0)
    
    return sampled_features


def sample_tokens_uniform(features, k, dim=1):
    """
    Uniformly sample k tokens from sequence features (evenly spaced).
    
    Args:
        features: Feature tensor of shape (batch_size, seq_len, hidden_dim)
        k: Number of tokens to sample
        dim: Dimension along which to sample (default: 1 for sequence dimension)
    
    Returns:
        Sampled features of shape (batch_size, k, hidden_dim)
    """
    batch_size, seq_len, hidden_dim = features.shape
    
    # If k >= seq_len, return all features
    if k >= seq_len:
        return features
    
    # Compute uniform indices
    indices = torch.linspace(0, seq_len - 1, k, device=features.device).long()
    
    # Sample features
    sampled_features = features[:, indices, :]
    
    return sampled_features


def sample_tokens_importance(features, k, importance_scores, dim=1):
    """
    Sample k tokens based on importance scores.
    
    Args:
        features: Feature tensor of shape (batch_size, seq_len, hidden_dim)
        k: Number of tokens to sample
        importance_scores: Importance scores of shape (batch_size, seq_len)
        dim: Dimension along which to sample (default: 1 for sequence dimension)
    
    Returns:
        Sampled features of shape (batch_size, k, hidden_dim)
    """
    batch_size, seq_len, hidden_dim = features.shape
    
    # If k >= seq_len, return all features
    if k >= seq_len:
        return features
    
    # Sample top-k indices based on importance scores
    _, top_indices = torch.topk(importance_scores, k, dim=dim)
    
    # Sort indices to maintain order
    top_indices, _ = torch.sort(top_indices, dim=dim)
    
    # Gather features
    sampled_features = torch.gather(
        features,
        dim=1,
        index=top_indices.unsqueeze(-1).expand(-1, -1, hidden_dim)
    )
    
    return sampled_features


def test_sampling():
    """Test token sampling functions."""
    batch_size = 4
    seq_len = 100
    hidden_dim = 768
    k = 50
    
    # Create dummy features
    features = torch.randn(batch_size, seq_len, hidden_dim)
    
    # Test random sampling
    sampled_random = sample_tokens(features, k)
    print(f"Random sampling: {features.shape} -> {sampled_random.shape}")
    
    # Test uniform sampling
    sampled_uniform = sample_tokens_uniform(features, k)
    print(f"Uniform sampling: {features.shape} -> {sampled_uniform.shape}")
    
    # Test importance-based sampling
    importance_scores = torch.rand(batch_size, seq_len)
    sampled_importance = sample_tokens_importance(features, k, importance_scores)
    print(f"Importance sampling: {features.shape} -> {sampled_importance.shape}")
    
    # Test with k >= seq_len
    sampled_all = sample_tokens(features, seq_len + 10)
    print(f"Sampling all (k >= seq_len): {features.shape} -> {sampled_all.shape}")


if __name__ == "__main__":
    test_sampling()
