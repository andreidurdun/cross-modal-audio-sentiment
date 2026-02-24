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
        
        print(f"✓ Loaded {len(self)} samples from {partition}")
        print(f"  Text EN embedding dim: {self.text_en_emb.shape[1]}")
        print(f"  Text ES embedding dim: {self.text_es_emb.shape[1]}")
        print(f"  Audio embedding dim: {self.audio_emb.shape[1]}")
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dict cu:
                - text_en_emb: (embedding_dim,)
                - text_es_emb: (embedding_dim,)
                - audio_emb: (embedding_dim,)
                - labels: (,)
                - file_id: str
        """
        return {
            'text_en_emb': self.text_en_emb[idx],
            'text_es_emb': self.text_es_emb[idx],
            'audio_emb': self.audio_emb[idx],
            'labels': self.labels[idx],
            'file_id': self.file_ids[idx],
        }
    
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


# ============================================================================
# EXEMPLU DE UTILIZARE
# ============================================================================

def example_usage():
    """Exemplu de utilizare a PrecomputedEmbeddingsDataset."""
    print("\n" + "="*80)
    print("EXEMPLU UTILIZARE: PrecomputedEmbeddingsDataset")
    print("="*80 + "\n")
    
    # 1. Creează dataset
    print("1. Creating dataset...")
    dataset = PrecomputedEmbeddingsDataset(
        embeddings_dir="data/embeddings",
        partition="train",
    )
    
    # 2. Verifică un sample
    print("\n2. Checking sample...")
    sample = dataset[0]
    print(f"   Text EN embedding: {sample['text_en_emb'].shape}")
    print(f"   Text ES embedding: {sample['text_es_emb'].shape}")
    print(f"   Audio embedding: {sample['audio_emb'].shape}")
    print(f"   Label: {sample['label']}")
    print(f"   File ID: {sample['file_id']}")
    
    # 3. Creează DataLoader
    print("\n3. Creating DataLoader...")
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # 4. Test batch
    print("\n4. Testing batch...")
    batch = next(iter(loader))
    print(f"   Batch text EN embeddings: {batch['text_en_emb'].shape}")
    print(f"   Batch text ES embeddings: {batch['text_es_emb'].shape}")
    print(f"   Batch audio embeddings: {batch['audio_emb'].shape}")
    print(f"   Batch labels: {batch['label'].shape}")
    
    # 5. Training loop example
    print("\n5. Example training loop (2 batches)...")
    
    from src.models import load_full_multimodal_model
    
    # Load model (fără backbones, doar fusion + CCMT)
    model = load_full_multimodal_model(
        freeze_backbones=True,
        projection_dim=0,  # Folosește dimensiunea nativă a embeddings-urilor
    )
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
    )
    
    criterion = torch.nn.CrossEntropyLoss()
    
    model.train()
    device = next(model.parameters()).device
    
    for i, batch in enumerate(loader):
        if i >= 2:  # Just 2 batches for demo
            break
        
        # Move to device
        text_en_emb = batch['text_en_emb'].to(device)
        text_es_emb = batch['text_es_emb'].to(device)
        audio_emb = batch['audio_emb'].to(device)
        labels = batch['label'].to(device)
        
        # Forward
        predictions = model(
            text_en_emb=text_en_emb,
            text_es_emb=text_es_emb,
            audio_emb=audio_emb,
        )
        
        # Loss
        loss = criterion(predictions, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"   Batch {i+1}: Loss = {loss.item():.4f}")
    
    print("\n" + "="*80)
    print("✅ EXAMPLE COMPLETED!")
    print("="*80 + "\n")


def example_create_all_dataloaders():
    """Exemplu: creează DataLoaders pentru toate partițiile."""
    print("\n" + "="*80)
    print("EXEMPLU: Create All DataLoaders")
    print("="*80 + "\n")
    
    dataloaders = create_dataloaders(
        embeddings_dir="data/embeddings",
        batch_size=32,
        num_workers=2,
    )
    
    print(f"\n✓ Created {len(dataloaders)} DataLoaders:")
    for partition, loader in dataloaders.items():
        print(f"  - {partition}: {len(loader.dataset)} samples, {len(loader)} batches")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test PrecomputedEmbeddingsDataset")
    parser.add_argument(
        '--example',
        type=str,
        default='usage',
        choices=['usage', 'dataloaders'],
        help='Which example to run',
    )
    
    args = parser.parse_args()
    
    if args.example == 'usage':
        example_usage()
    elif args.example == 'dataloaders':
        example_create_all_dataloaders()
