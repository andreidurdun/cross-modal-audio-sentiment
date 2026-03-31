"""
Dataset Loader pentru embeddings pre-calculate.
Permite training rapid folosind embeddings salvate pe disc.

Usage:
    from scripts.precomputed_embeddings_dataset import PrecomputedEmbeddingsDataset
    
    dataset = PrecomputedEmbeddingsDataset('data/embeddings', 'train')
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Optional


class PrecomputedEmbeddingsDataset(Dataset):
    """
    Dataset pentru embeddings pre-calculate.
    Încarcă embeddings de pe disc pentru training rapid.
    """
    
    def __init__(
        self,
        embeddings_dir: str = "data/embeddings",
        partition: str = "train",
        device: str = "cpu",
        regression: bool = False,
    ):
        """
        Args:
            embeddings_dir: Director cu embeddings salvate
            partition: 'train', 'val', sau 'test1'
            device: Device unde să fie încărcate embeddings
        """
        self.embeddings_dir = Path(embeddings_dir)
        self.partition = partition
        self.device = device
        self.regression = regression
        # Încarcă embeddings
        embeddings_file = self.embeddings_dir / f"embeddings_{partition}.pt"
        if not embeddings_file.exists():
            raise FileNotFoundError(
                f"Embeddings file not found: {embeddings_file}\n"
                f"Run 'python scripts/extract_and_save_embeddings.py --partition {partition}' first!"
            )
        print(f"Loading embeddings from {embeddings_file}...")
        self.data = torch.load(embeddings_file, map_location='cpu')
        self.text_en_emb = self.data['text_en']
        self.text_es_emb = self.data['text_es']
        self.audio_emb = self.data['audio']
        self.labels = self.data['labels']
        self.file_ids = self.data['file_ids']
        self.metadata = self.data.get('metadata', {})
        # Pentru regresie: încearcă să încarci valence/arousal dacă există
        self.valence = self.data.get('valence', None)
        self.arousal = self.data.get('arousal', None)
        if self.regression:
            if self.valence is None or self.arousal is None:
                raise ValueError("Embeddings file must contain 'valence' and 'arousal' tensors for regression task.")
            self.valence = self.valence.to(torch.float32)
            self.arousal = self.arousal.to(torch.float32)
            # --- NORMALIZARE MIN-MAX [0, 1] ---
            # Setăm limitele teoretice ale setului MSP-Podcast (1.0 și 7.0)
            # Este important să folosim limitele globale, nu min/max din batch-ul curent,
            # pentru a păstra consistența între Train, Val și Test.
            MIN_VAL = 1.0
            MAX_VAL = 7.0
            # Formula: (X - X_min) / (X_max - X_min)
            self.valence = (self.valence - MIN_VAL) / (MAX_VAL - MIN_VAL)
            self.arousal = (self.arousal - MIN_VAL) / (MAX_VAL - MIN_VAL)
        print(f"✓ Loaded {len(self)} samples from {partition}")
        print(f"  Text EN embedding dim: {self.text_en_emb.shape[1]}")
        print(f"  Text ES embedding dim: {self.text_es_emb.shape[1]}")
        print(f"  Audio embedding dim: {self.audio_emb.shape[1]}")

    @staticmethod
    def denormalize_val_arousal(val_arousal_norm: torch.Tensor) -> torch.Tensor:
        """
        Denormalize valence/arousal from [0, 1] back to [1, 7] scale.
        Args:
            val_arousal_norm: Tensor of shape (..., 2) with normalized values
        Returns:
            Tensor with denormalized values
        """
        MIN_VAL = 1.0
        MAX_VAL = 7.0
        return val_arousal_norm * (MAX_VAL - MIN_VAL) + MIN_VAL
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        batch = {
            'text_en_emb': self.text_en_emb[idx],
            'text_es_emb': self.text_es_emb[idx],
            'audio_emb': self.audio_emb[idx],
            'labels': self.labels[idx],
            'file_id': self.file_ids[idx],
        }
        if self.regression:
            batch['val_arousal'] = torch.stack([
                self.valence[idx],
                self.arousal[idx]
            ])
        return batch
    
    def get_embedding_dims(self) -> Dict[str, int]:
        """Returnează dimensiunile embeddings."""
        return {
            'text_en': self.text_en_emb.shape[1],
            'text_es': self.text_es_emb.shape[1],
            'audio': self.audio_emb.shape[1],
        }


def create_dataloaders(
    embeddings_dir: str = "data/embeddings",
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Dict[str, DataLoader]:
    """
    Creează DataLoaders pentru toate partițiile.
    
    Args:
        embeddings_dir: Director cu embeddings
        batch_size: Batch size
        num_workers: Număr workers pentru DataLoader
        pin_memory: Pin memory pentru GPU
    
    Returns:
        Dict cu DataLoaders pentru 'train', 'val', 'test'
    """
    dataloaders = {}
    
    for partition in ['train', 'val', 'test1']:
        try:
            dataset = PrecomputedEmbeddingsDataset(embeddings_dir, partition)
            
            dataloaders[partition] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(partition == 'train'),
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            
            print(f"✓ Created DataLoader for {partition}: {len(dataset)} samples, "
                  f"{len(dataloaders[partition])} batches")
        
        except FileNotFoundError as e:
            print(f"⚠ Skipping {partition}: {e}")
    
    return dataloaders

