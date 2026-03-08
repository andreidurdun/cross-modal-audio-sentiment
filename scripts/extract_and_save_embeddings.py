import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import pickle
from typing import Dict, List, Optional
from datetime import datetime

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoFeatureExtractor

from src.models import load_all_backbones
from src.data.dataset import MSP_Podcast_Dataset


class EmbeddingExtractor:
    
    def __init__(
        self,
        output_dir: str = "MSP_Podcast/embeddings",
        text_en_checkpoint: str = "checkpoints/roberta_text_en",
        text_es_checkpoint: str = "checkpoints/roberta_text_es",
        audio_checkpoint: str = "checkpoints/wavlm_audio",
        dataset_root: str = "MSP_Podcast",
        projection_dim: Optional[int] = None,
        device: str = "cuda",
        batch_size: int = 32,
    ):
        """
        Args:
            output_dir: Director unde se salvează embeddings
            text_en_checkpoint: Path la checkpoint RoBERTa EN
            text_es_checkpoint: Path la checkpoint RoBERTa ES
            audio_checkpoint: Path la checkpoint WavLM
            projection_dim: Dimensiune proiecție (None = dimensiune nativă)
            device: Device pentru procesare
            batch_size: Batch size pentru procesare
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.batch_size = batch_size
        self.projection_dim = projection_dim
        self.dataset_root = Path(dataset_root)
        
        print("\n" + "="*80)
        print("EMBEDDING EXTRACTOR INITIALIZATION")
        print("="*80)
        
        # Încarcă backbones
        print("\nLoading backbones...")
        self.backbones = load_all_backbones(
            text_en_checkpoint=text_en_checkpoint,
            text_es_checkpoint=text_es_checkpoint,
            audio_checkpoint=audio_checkpoint,
            freeze=True,
            projection_dim=projection_dim,
        )
        
        # Move to device
        for name, backbone in self.backbones.items():
            self.backbones[name] = backbone.to(device).eval()
        
        # Audio feature extractor
        self.audio_feature_extractor = AutoFeatureExtractor.from_pretrained(
            "microsoft/wavlm-base-plus"
        )
        
        print(f"\nBackbones loaded and ready!")
        print(f"  Device: {device}")
        print(f"  Batch size: {batch_size}")
        print(f"  Output dir: {self.output_dir}")
        print("="*80 + "\n")
    
    def extract_embeddings_for_partition(
        self,
        partition: str,
        save_individually: bool = False,
    ) -> Dict:
        """
        Extrage embeddings pentru o partiție specifică.
        
        Args:
            partition: 'train', 'val', 'test1', 'test2', sau 'dev'
            save_individually: Dacă True, salvează fiecare sample individual
        
        Returns:
            Dict cu embeddings și metadata
        """
        print(f"\n{'='*80}")
        print(f"EXTRACTING EMBEDDINGS FOR PARTITION: {partition.upper()}")
        print(f"{'='*80}\n")
        
        # Încarcă dataset
        print(f"Loading MSP-Podcast {partition} dataset...")
        data_dir = self.dataset_root
        labels_csv = data_dir / "Labels" / "labels_consensus.csv"
        transcripts_en_json = data_dir / "Transcription_en.json"
        transcripts_es_json = data_dir / "Transcription_es.json"
        audio_root = data_dir / "Audios"

        if not labels_csv.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_csv}")
        if not transcripts_en_json.exists():
            raise FileNotFoundError(f"Transcripts JSON not found: {transcripts_en_json}")
        if not transcripts_es_json.exists():
            raise FileNotFoundError(f"Transcripts JSON not found: {transcripts_es_json}")
        if not audio_root.exists():
            raise FileNotFoundError(f"Audio directory not found: {audio_root}")

        partition_map = {
            "train": "Train",
            "val": "Development",
            "dev": "Development",
            "test1": "Test1",
            "test2": "Test2",
        }
        dataset_partition = partition_map.get(partition.lower(), partition)

        dataset = MSP_Podcast_Dataset(
            audio_root=str(audio_root),
            labels_csv=str(labels_csv),
            transcripts_en_json=str(transcripts_en_json),
            transcripts_es_json=str(transcripts_es_json),
            partition=dataset_partition,
            modalities=["audio", "text_en", "text_es"],
        )
        print(f"Loaded {len(dataset)} samples\n")
        if len(dataset) == 0:
            raise ValueError(
                f"No samples for partition '{partition}' (mapped to '{dataset_partition}'). "
                "Check Split_Set values in labels_consensus.csv."
            )
        
        # Storage
        all_embeddings = {
            'text_en': [],
            'text_es': [],
            'audio': [],
            'labels': [],
            'file_ids': [],
        }
        
        # Process în batches
        num_batches = (len(dataset) + self.batch_size - 1) // self.batch_size
        
        with torch.no_grad():
            for batch_idx in tqdm(range(num_batches), desc=f"Processing {partition}"):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, len(dataset))
                
                # Collect batch data
                batch_texts_en = []
                batch_texts_es = []
                batch_audios = []
                batch_labels = []
                batch_file_ids = []
                
                for idx in range(start_idx, end_idx):
                    sample = dataset[idx]
                    batch_texts_en.append(sample['text_en'])
                    batch_texts_es.append(sample['text_es'])
                    
                    # Process audio
                    audio = sample['audio'].numpy() if isinstance(sample['audio'], torch.Tensor) else sample['audio']
                    
                    # Clean audio
                    if np.isnan(audio).any() or np.isinf(audio).any():
                        audio = np.zeros_like(audio)
                    
                    # Truncate to 6 seconds
                    max_samples = 16000 * 6
                    if audio.shape[-1] > max_samples:
                        audio = audio[..., :max_samples]
                    
                    batch_audios.append(audio)
                    batch_labels.append(sample['label'])
                    batch_file_ids.append(sample.get('file_id', f'{partition}_{idx}'))
                
                # Extract text embeddings
                text_en_emb = self.backbones['text_en'](batch_texts_en).cpu()
                text_es_emb = self.backbones['text_es'](batch_texts_es).cpu()
                
                # Extract audio embeddings
                audio_emb = self._extract_audio_embeddings_batch(batch_audios)

                #Aplicăm strict lungimea de 100 tokeni înainte să le mutăm pe CPU
                text_en_emb = self._pad_or_truncate(text_en_emb, target_len=100).cpu()
                text_es_emb = self._pad_or_truncate(text_es_emb, target_len=100).cpu()
                audio_emb = self._pad_or_truncate(audio_emb, target_len=100).cpu()
                
                # Store embeddings
                all_embeddings['text_en'].append(text_en_emb)
                all_embeddings['text_es'].append(text_es_emb)
                all_embeddings['audio'].append(audio_emb)
                all_embeddings['labels'].extend(batch_labels)
                all_embeddings['file_ids'].extend(batch_file_ids)
                
                # Optionally save individual samples
                if save_individually:
                    self._save_batch_individually(
                        batch_idx,
                        partition,
                        text_en_emb,
                        text_es_emb,
                        audio_emb,
                        batch_labels,
                        batch_file_ids,
                    )
        
        # Concatenate all embeddings
        embeddings_dict = {
            'text_en': torch.cat(all_embeddings['text_en'], dim=0),
            'text_es': torch.cat(all_embeddings['text_es'], dim=0),
            'audio': torch.cat(all_embeddings['audio'], dim=0),
            'labels': torch.tensor(all_embeddings['labels'], dtype=torch.long),
            'file_ids': all_embeddings['file_ids'],
            'metadata': {
                'partition': partition,
                'num_samples': len(dataset),
                'projection_dim': self.projection_dim,
                'embedding_dims': {
                    'text_en': all_embeddings['text_en'][0].shape[1],
                    'text_es': all_embeddings['text_es'][0].shape[1],
                    'audio': all_embeddings['audio'][0].shape[1],
                },
                'extraction_date': datetime.now().isoformat(),
            }
        }
        
        # Save consolidated embeddings
        self._save_embeddings(embeddings_dict, partition)
        
        return embeddings_dict
    
    def _extract_audio_embeddings_batch(self, audios: List[np.ndarray]) -> torch.Tensor:
        """Extrage embeddings audio pentru un batch."""
        # Process audio cu feature extractor
        audio_inputs = self.audio_feature_extractor(
            audios,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        
        # Move to device
        audio_inputs = {k: v.to(self.device) for k, v in audio_inputs.items()}
        
        # Extract embeddings
        audio_emb = self.backbones['audio'](
            audio_inputs['input_values'],
            attention_mask=audio_inputs.get('attention_mask'),
        )
        
        return audio_emb.cpu()
    
    def _pad_or_truncate(self, tensor: torch.Tensor, target_len: int = 100) -> torch.Tensor:
        """
        Asigură că tensorul are dimensiunea [batch_size, target_len, embed_dim].
        Păstrează mereu tokenul de la indexul 0 (ex: [CLS] pentru text).
        """
        if tensor.ndim == 2:
            return tensor

        batch_size, seq_len, embed_dim = tensor.shape
        
        if seq_len == target_len:
            return tensor
            
        elif seq_len > target_len:
            # Tăiem, dar păstrăm ordinea de la început (inclusiv indexul 0)
            return tensor[:, :target_len, :]
            
        else:
            # Pad cu zerouri
            padding = torch.zeros((batch_size, target_len - seq_len, embed_dim), device=tensor.device, dtype=tensor.dtype)
            return torch.cat([tensor, padding], dim=1)
    
    def _save_embeddings(self, embeddings_dict: Dict, partition: str):
        """Salvează embeddings consolidate."""
        output_file = self.output_dir / f"embeddings_{partition}.pt"
        
        print(f"\n{'='*80}")
        print(f"SAVING EMBEDDINGS: {partition}")
        print(f"{'='*80}")
        
        # Save as PyTorch tensors
        torch.save(embeddings_dict, output_file)
        
        print(f"\n✓ Saved to: {output_file}")
        print(f"  Text EN shape: {embeddings_dict['text_en'].shape}")
        print(f"  Text ES shape: {embeddings_dict['text_es'].shape}")
        print(f"  Audio shape: {embeddings_dict['audio'].shape}")
        print(f"  Labels shape: {embeddings_dict['labels'].shape}")
        print(f"  File size: {output_file.stat().st_size / (1024**2):.2f} MB")
        
        # Save metadata separately
        metadata_file = self.output_dir / f"metadata_{partition}.json"
        with open(metadata_file, 'w') as f:
            json.dump(embeddings_dict['metadata'], f, indent=2)
        
        print(f"  Metadata: {metadata_file}")
        print("="*80 + "\n")
    
    def _save_batch_individually(
        self,
        batch_idx: int,
        partition: str,
        text_en_emb: torch.Tensor,
        text_es_emb: torch.Tensor,
        audio_emb: torch.Tensor,
        labels: List[int],
        file_ids: List[str],
    ):
        """Salvează fiecare sample din batch individual (opțional)."""
        individual_dir = self.output_dir / "individual" / partition
        individual_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(len(labels)):
            sample_data = {
                'text_en_emb': text_en_emb[i],
                'text_es_emb': text_es_emb[i],
                'audio_emb': audio_emb[i],
                'label': labels[i],
                'file_id': file_ids[i],
            }
            
            sample_file = individual_dir / f"{file_ids[i]}.pt"
            torch.save(sample_data, sample_file)
    
    def extract_all_partitions(self, save_individually: bool = False):
        """Extrage embeddings pentru toate partițiile."""
        partitions = ['train', 'val', 'test1']
        
        print("\n" + "="*80)
        print("EXTRACTING ALL PARTITIONS")
        print("="*80)
        print(f"Partitions: {partitions}")
        print(f"Save individually: {save_individually}")
        print("="*80 + "\n")
        
        results = {}
        
        for partition in partitions:
            try:
                embeddings_dict = self.extract_embeddings_for_partition(
                    partition=partition,
                    save_individually=save_individually,
                )
                results[partition] = embeddings_dict['metadata']
            except Exception as e:
                print(f"\n⚠ Error processing {partition}: {e}")
                import traceback
                traceback.print_exc()
        
        # Save summary
        summary_file = self.output_dir / "extraction_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n" + "="*80)
        print("EXTRACTION COMPLETE")
        print("="*80)
        print(f"Summary saved to: {summary_file}")
        print(f"Total partitions processed: {len(results)}")
        print("="*80 + "\n")


class EmbeddingLoader:
    """Utilitar pentru încărcarea embedding-urilor salvate."""
    
    def __init__(self, embeddings_dir: str = "data/embeddings"):
        self.embeddings_dir = Path(embeddings_dir)
    
    def load_partition(self, partition: str) -> Dict:
        """Încarcă embeddings pentru o partiție."""
        embeddings_file = self.embeddings_dir / f"embeddings_{partition}.pt"
        
        if not embeddings_file.exists():
            raise FileNotFoundError(f"Embeddings not found: {embeddings_file}")
        
        embeddings_dict = torch.load(embeddings_file)
        
        print(f"\n✓ Loaded embeddings for {partition}")
        print(f"  Text EN: {embeddings_dict['text_en'].shape}")
        print(f"  Text ES: {embeddings_dict['text_es'].shape}")
        print(f"  Audio: {embeddings_dict['audio'].shape}")
        print(f"  Labels: {embeddings_dict['labels'].shape}")
        
        return embeddings_dict
    
    def load_all_partitions(self) -> Dict[str, Dict]:
        """Încarcă embeddings pentru toate partițiile disponibile."""
        partitions = ['train', 'val', 'test1']
        all_embeddings = {}
        
        for partition in partitions:
            try:
                all_embeddings[partition] = self.load_partition(partition)
            except FileNotFoundError:
                print(f"⚠ Skipping {partition} (not found)")
        
        return all_embeddings


def main():
    parser = argparse.ArgumentParser(description="Extract and save embeddings from backbones")
    parser.add_argument(
        '--partition',
        type=str,
        default='train, val',
        help='Partition to process (default: all). Supports comma-separated values.',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='MSP_Podcast/embeddings',
        help='Output directory for embeddings',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for processing (default: 32)',
    )
    parser.add_argument(
        '--projection-dim',
        type=int,
        default=0,
        help='Projection dimension for embeddings (default: 768, use 0 for native)',
    )
    parser.add_argument(
        '--save-individually',
        action='store_true',
        help='Save each sample individually (in addition to consolidated file)',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (default: cuda if available)',
    )
    
    args = parser.parse_args()
    
    # Convert projection_dim
    projection_dim = args.projection_dim if args.projection_dim > 0 else None
    
    print("\n" + "="*80)
    print("EMBEDDING EXTRACTION SCRIPT")
    print("="*80)
    print(f"Partition: {args.partition}")
    print(f"Output dir: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Projection dim: {projection_dim}")
    print(f"Device: {args.device}")
    print(f"Save individually: {args.save_individually}")
    print("="*80 + "\n")
    
    # Create extractor
    extractor = EmbeddingExtractor(
        output_dir=args.output_dir,
        projection_dim=projection_dim,
        device=args.device,
        batch_size=args.batch_size,
    )
    
    # Extract embeddings
    allowed_partitions = {'train', 'val', 'test1', 'test2', 'dev', 'all'}
    if args.partition == 'all':
        extractor.extract_all_partitions(save_individually=args.save_individually)
    else:
        partitions = [p.strip() for p in args.partition.split(',') if p.strip()]
        invalid = [p for p in partitions if p not in allowed_partitions]
        if invalid:
            raise ValueError(
                f"Invalid partition(s): {', '.join(invalid)}. "
                f"Allowed: {sorted(allowed_partitions)}"
            )
        for partition in partitions:
            extractor.extract_embeddings_for_partition(
                partition=partition,
                save_individually=args.save_individually,
            )
    
    print("\nEmbedding extraction completed successfully!")
    print(f" Embeddings saved to: {args.output_dir}\n")


if __name__ == "__main__":
    main()
