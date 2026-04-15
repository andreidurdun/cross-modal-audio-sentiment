"""Compatibility wrapper for dataset imports kept under scripts/."""

try:
    from scripts._bootstrap import project_root
except ModuleNotFoundError:
    from _bootstrap import project_root

project_root()

from src.data.precomputed_embeddings_dataset import ( 
    PrecomputedEmbeddingsDataset,
    create_dataloaders,
)

__all__ = ["PrecomputedEmbeddingsDataset", "create_dataloaders"]