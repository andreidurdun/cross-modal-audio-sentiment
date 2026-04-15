import sys
from pathlib import Path

try:
    from scripts._bootstrap import project_root
except ModuleNotFoundError:
    from _bootstrap import project_root

PROJECT_ROOT = project_root()

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


SUPPORTED_MODALITIES = ["text_en", "text_es", "text_de", "text_fr", "audio"]
TRANSCRIPT_FILES = {
    "text_en": "Transcription_en.json",
    "text_es": "Transcription_es.json",
    "text_de": "Transcription_de.json",
    "text_fr": "Transcription_fr.json",
}


def parse_modalities(modalities_arg: str | None) -> list[str]:
    if not modalities_arg:
        return ["text_en", "text_es", "audio"]

    modalities = [item.strip() for item in modalities_arg.split(",") if item.strip()]
    invalid_modalities = [item for item in modalities if item not in SUPPORTED_MODALITIES]
    if invalid_modalities:
        raise ValueError(
            f"Modalitati invalide: {invalid_modalities}. Alege dintre {SUPPORTED_MODALITIES}"
        )
    return modalities


def normalize_checkpoint_dir(checkpoint_path: str) -> str:
    checkpoint = Path(checkpoint_path)
    if checkpoint.name == "best_model":
        return str(checkpoint.parent)
    return str(checkpoint)


def build_modality_suffix(modalities: list[str]) -> str:
    return "_".join(modalities)


def resolve_output_dir(output_dir_arg: str | None, modalities: list[str]) -> str:
    if output_dir_arg:
        return output_dir_arg

    if modalities == ["text_en", "text_es", "audio"]:
        return "MSP_Podcast/embeddings"

    return f"MSP_Podcast/embeddings_{build_modality_suffix(modalities)}"


def ensure_output_dir_available(output_dir: Path, allow_overwrite: bool) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"Output path exists and is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()) and not allow_overwrite:
        raise FileExistsError(
            f"Output directory already exists and is not empty: {output_dir}. "
            "Use a different --output-dir or pass --allow-overwrite explicitly."
        )


def parse_reuse_dirs(reuse_from_arg: str | None) -> list[str]:
    if not reuse_from_arg:
        return []
    return [item.strip() for item in reuse_from_arg.split(';') if item.strip()]


class EmbeddingExtractor:
    
    def __init__(
        self,
        output_dir: str = "MSP_Podcast/embeddings",
        text_en_checkpoint: str = "checkpoints/roberta_text_en",
        text_es_checkpoint: str = "checkpoints/roberta_text_es",
        text_de_checkpoint: str = "checkpoints/roberta_text_de",
        text_fr_checkpoint: str = "checkpoints/roberta_text_fr",
        audio_checkpoint: str = "checkpoints/wavlm_audio",
        dataset_root: str = "MSP_Podcast",
        projection_dim: Optional[int] = None,
        device: str = "cuda",
        batch_size: int = 32,
        modalities: Optional[List[str]] = None,
        reuse_from_dirs: Optional[List[str]] = None,
    ):
        """
        Args:
            output_dir: Director unde se salveaza embeddings
            text_en_checkpoint: Path la checkpoint RoBERTa EN
            text_es_checkpoint: Path la checkpoint RoBERTa ES
            audio_checkpoint: Path la checkpoint WavLM
            projection_dim: Dimensiune proiectie (None = dimensiune nativa)
            device: Device pentru procesare
            batch_size: Batch size pentru procesare
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.batch_size = batch_size
        self.projection_dim = projection_dim
        self.dataset_root = Path(dataset_root)
        self.modalities = list(modalities or ["text_en", "text_es", "audio"])
        self.reuse_from_dirs = [Path(path) for path in (reuse_from_dirs or [])]
        
        print("\n" + "="*80)
        print("EMBEDDING EXTRACTOR INITIALIZATION")
        print("="*80)
        
        # Incarca backbones
        print("\nLoading backbones...")
        self.backbones = load_all_backbones(
            text_en_checkpoint=text_en_checkpoint,
            text_es_checkpoint=text_es_checkpoint,
            text_de_checkpoint=text_de_checkpoint,
            text_fr_checkpoint=text_fr_checkpoint,
            audio_checkpoint=audio_checkpoint,
            freeze=True,
            projection_dim=projection_dim,
            modalities=self.modalities,
        )
        
        # Move to device
        for name, backbone in self.backbones.items():
            self.backbones[name] = backbone.to(device).eval()
        
        # Audio feature extractor
        self.audio_feature_extractor = None
        if "audio" in self.modalities:
            self.audio_feature_extractor = AutoFeatureExtractor.from_pretrained(
                "microsoft/wavlm-base-plus"
            )
        
        print(f"\nBackbones loaded and ready!")
        print(f"  Modalities: {self.modalities}")
        print(f"  Device: {device}")
        print(f"  Batch size: {batch_size}")
        print(f"  Output dir: {self.output_dir}")
        if self.reuse_from_dirs:
            print(f"  Reuse from dirs: {[str(path) for path in self.reuse_from_dirs]}")
        print("="*80 + "\n")
    
    def _load_reusable_partition_embeddings(
        self,
        partition: str,
        file_ids: list[str],
    ) -> dict[str, torch.Tensor]:
        reusable: dict[str, torch.Tensor] = {}
        if not self.reuse_from_dirs:
            return reusable

        target_file_ids = [str(file_id) for file_id in file_ids]
        for reuse_dir in self.reuse_from_dirs:
            embeddings_path = Path(reuse_dir) / f"embeddings_{partition}.pt"
            if not embeddings_path.exists():
                continue

            data = torch.load(embeddings_path, map_location="cpu")
            source_file_ids = [str(file_id) for file_id in data.get("file_ids", [])]
            if not source_file_ids:
                continue

            index_lookup = {file_id: idx for idx, file_id in enumerate(source_file_ids)}
            if any(file_id not in index_lookup for file_id in target_file_ids):
                print(f"Skipping reuse from {embeddings_path} because file_ids do not align.")
                continue

            selection = torch.tensor([index_lookup[file_id] for file_id in target_file_ids], dtype=torch.long)
            reused_here = []
            for modality in self.modalities:
                if modality in reusable or modality not in data:
                    continue
                tensor = data[modality]
                if isinstance(tensor, torch.Tensor) and tensor.shape[0] == len(source_file_ids):
                    reusable[modality] = tensor.index_select(0, selection)
                    reused_here.append(modality)

            if reused_here:
                print(f"Reusing modalities {reused_here} from {embeddings_path}")

            if len(reusable) == len(self.modalities):
                break

        return reusable

    def extract_embeddings_for_partition(
        self,
        partition: str,
        save_individually: bool = False,
    ) -> Dict:
        """
        Extrage embeddings pentru o partitie specifica.
        
        Args:
            partition: 'train', 'val', 'test1', 'test2', sau 'dev'
            save_individually: Daca True, salveaza fiecare sample individual
        
        Returns:
            Dict cu embeddings si metadata
        """
        print(f"\n{'='*80}")
        print(f"EXTRACTING EMBEDDINGS FOR PARTITION: {partition.upper()}")
        print(f"{'='*80}\n")
        
        # Incarca dataset
        print(f"Loading MSP-Podcast {partition} dataset...")
        data_dir = self.dataset_root
        labels_csv = data_dir / "Labels" / "labels_consensus.csv"
        audio_root = data_dir / "Audios"

        if not labels_csv.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_csv}")
        transcript_paths = {}
        for modality in self.modalities:
            if not modality.startswith("text_"):
                continue
            transcript_path = data_dir / TRANSCRIPT_FILES[modality]
            if not transcript_path.exists():
                raise FileNotFoundError(f"Transcripts JSON not found: {transcript_path}")
            transcript_paths[modality] = transcript_path
        if "audio" in self.modalities and not audio_root.exists():
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
            transcripts_en_json=str(transcript_paths["text_en"]) if "text_en" in transcript_paths else None,
            transcripts_es_json=str(transcript_paths["text_es"]) if "text_es" in transcript_paths else None,
            transcripts_de_json=str(transcript_paths["text_de"]) if "text_de" in transcript_paths else None,
            transcripts_fr_json=str(transcript_paths["text_fr"]) if "text_fr" in transcript_paths else None,
            partition=dataset_partition,
            modalities=self.modalities,
        )
        print(f"Loaded {len(dataset)} samples\n")
        if len(dataset) == 0:
            raise ValueError(
                f"No samples for partition '{partition}' (mapped to '{dataset_partition}'). "
                "Check Split_Set values in labels_consensus.csv."
            )
        
        partition_file_ids = [str(file_id) for file_id in dataset.metadata['FileName'].tolist()]
        reusable_embeddings = self._load_reusable_partition_embeddings(partition, partition_file_ids)

        # Storage
        all_embeddings = {
            **{modality: [] for modality in self.modalities},
            'labels': [],
            'file_ids': [],
            'valence': [],
            'arousal': [],
        }
        
        # Process in batches
        num_batches = (len(dataset) + self.batch_size - 1) // self.batch_size
        text_modalities_to_compute = [
            modality for modality in self.modalities
            if modality.startswith("text_") and modality not in reusable_embeddings
        ]
        audio_needs_compute = "audio" in self.modalities and "audio" not in reusable_embeddings

        with torch.no_grad():
            for batch_idx in tqdm(range(num_batches), desc=f"Processing {partition}"):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, len(dataset))
                
                # Collect batch data
                batch_texts = {modality: [] for modality in text_modalities_to_compute}
                batch_audios = []
                batch_labels = []
                batch_file_ids = []
                
                for idx in range(start_idx, end_idx):
                    row = dataset.metadata.iloc[idx]
                    file_id = str(row['FileName'])
                    file_id_key = str(row['file_id_key'])

                    for modality in batch_texts:
                        batch_texts[modality].append(dataset.transcript_caches[modality].get(file_id_key, ""))

                    if audio_needs_compute:
                        sample = dataset[idx]
                        audio = sample['audio'].numpy() if isinstance(sample['audio'], torch.Tensor) else sample['audio']
                        if np.isnan(audio).any() or np.isinf(audio).any():
                            audio = np.zeros_like(audio)

                        max_samples = 16000 * 6
                        if audio.shape[-1] > max_samples:
                            audio = audio[..., :max_samples]

                        batch_audios.append(audio)

                    batch_labels.append(int(row['label_id']))
                    batch_file_ids.append(file_id)
                    all_embeddings['valence'].append(float(row['EmoVal']))
                    all_embeddings['arousal'].append(float(row['EmoAct']))
                
                batch_embeddings = {}
                for modality in self.modalities:
                    if modality in reusable_embeddings:
                        batch_embeddings[modality] = reusable_embeddings[modality][start_idx:end_idx].cpu()
                        continue

                    if modality.startswith("text_"):
                        batch_embeddings[modality] = self._pad_or_truncate(
                            self.backbones[modality](batch_texts[modality]),
                            target_len=100,
                        ).cpu()

                if audio_needs_compute:
                    batch_embeddings['audio'] = self._pad_or_truncate(
                        self._extract_audio_embeddings_batch(batch_audios),
                        target_len=100,
                    ).cpu()

                for modality, embeddings in batch_embeddings.items():
                    all_embeddings[modality].append(embeddings)
                all_embeddings['labels'].extend(batch_labels)
                all_embeddings['file_ids'].extend(batch_file_ids)
                
                # Optionally save individual samples
                if save_individually:
                    self._save_batch_individually(
                        batch_idx,
                        partition,
                        batch_embeddings,
                        batch_labels,
                        batch_file_ids,
                    )
        
        # Concatenate all embeddings
        embeddings_dict = {
            **{
                modality: torch.cat(all_embeddings[modality], dim=0)
                for modality in self.modalities
            },
            'labels': torch.tensor(all_embeddings['labels'], dtype=torch.long),
            'file_ids': all_embeddings['file_ids'],
            'valence': torch.tensor(all_embeddings['valence'], dtype=torch.float32),
            'arousal': torch.tensor(all_embeddings['arousal'], dtype=torch.float32),
            'metadata': {
                'partition': partition,
                'num_samples': len(dataset),
                'projection_dim': self.projection_dim,
                'modalities': self.modalities,
                'embedding_dims': {
                    modality: all_embeddings[modality][0].shape[-1]
                    for modality in self.modalities
                },
                'extraction_date': datetime.now().isoformat(),
            }
        }
        
        # Save consolidated embeddings
        self._save_embeddings(embeddings_dict, partition)
        
        return embeddings_dict
    
    def _extract_audio_embeddings_batch(self, audios: List[np.ndarray]) -> torch.Tensor:
        """Extrage embeddings audio pentru un batch."""
        if self.audio_feature_extractor is None:
            raise ValueError("Audio feature extractor nu este disponibil deoarece modalitatea 'audio' nu este activa")

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
        Asigura ca tensorul are dimensiunea [batch_size, target_len, embed_dim].
        Pastreaza mereu tokenul de la indexul 0 (ex: [CLS] pentru text).
        """
        if tensor.ndim == 2:
            return tensor

        batch_size, seq_len, embed_dim = tensor.shape
        
        if seq_len == target_len:
            return tensor
            
        elif seq_len > target_len:
            # Taiem, dar pastram ordinea de la inceput (inclusiv indexul 0)
            return tensor[:, :target_len, :]
            
        else:
            # Pad cu zerouri
            padding = torch.zeros((batch_size, target_len - seq_len, embed_dim), device=tensor.device, dtype=tensor.dtype)
            return torch.cat([tensor, padding], dim=1)
    
    def _save_embeddings(self, embeddings_dict: Dict, partition: str):
        """Salveaza embeddings consolidate."""
        output_file = self.output_dir / f"embeddings_{partition}.pt"
        
        print(f"\n{'='*80}")
        print(f"SAVING EMBEDDINGS: {partition}")
        print(f"{'='*80}")
        
        # Save as PyTorch tensors
        torch.save(embeddings_dict, output_file)
        
        print(f"\n[OK] Saved to: {output_file}")
        for modality in self.modalities:
            print(f"  {modality} shape: {embeddings_dict[modality].shape}")
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
        batch_embeddings: Dict[str, torch.Tensor],
        labels: List[int],
        file_ids: List[str],
    ):
        """Salveaza fiecare sample din batch individual (optional)."""
        individual_dir = self.output_dir / "individual" / partition
        individual_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(len(labels)):
            sample_data = {
                'label': labels[i],
                'file_id': file_ids[i],
            }
            for modality, embeddings in batch_embeddings.items():
                sample_data[f'{modality}_emb'] = embeddings[i]
            
            sample_file = individual_dir / f"{file_ids[i]}.pt"
            torch.save(sample_data, sample_file)
    
    def extract_all_partitions(self, save_individually: bool = False):
        """Extrage embeddings pentru toate partitiile."""
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
                print(f"\n[!] Error processing {partition}: {e}")
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

    def add_german_embeddings(
        self,
        partition: str,
        checkpoint_path: str = "checkpoints/roberta_text_de",
        lang_key: str = "text_de"
    ):
        """
        Adauga embeddings pentru limba germana intr-un fisier .pt deja existent,
        fara a recalcula audio sau engleza.
        """
        existing_file = self.output_dir / f"embeddings_{partition}.pt"
        if not existing_file.exists():
            raise FileNotFoundError(f"Nu exista fisierul de baza: {existing_file}. Ruleaza extragerea initiala mai intai.")
            
        print(f"\n{'='*80}")
        print(f"ADDING {lang_key.upper()} EMBEDDINGS TO: {partition.upper()}")
        print(f"{'='*80}")
        
        print("Incarcam embedding-urile existente...")
        embeddings_dict = torch.load(existing_file)
        
        if lang_key in embeddings_dict:
            print(f"Atentie: Cheia '{lang_key}' exista deja in fisier. Va fi suprascrisa.")
            
        print(f"Incarcam modelul pentru {lang_key} din {checkpoint_path}...")
        try:
            from transformers import AutoModel
            from peft import PeftModel
            # Incarcam GBERT + LoRA
            base_model_name = "deepset/gbert-base"
            print(f"Incarcam modelul de baza: {base_model_name}...")
            base_model = AutoModel.from_pretrained(base_model_name)
            
            print(f"Atasam adaptoarele LoRA din: {checkpoint_path}...")
            de_backbone = PeftModel.from_pretrained(base_model, checkpoint_path)
            # Mutam pe placa video
            de_backbone = de_backbone.to(self.device).eval()
        except Exception as e:
            print(f"Eroare la incarcarea modelului PEFT: {e}")
            return

        # Pregatim datasetul pentru germana
        data_dir = self.dataset_root
        labels_csv = data_dir / "Labels" / "labels_consensus.csv"
        transcripts_de_json = data_dir / "Transcription_de.json"
        
        partition_map = {"train": "Train", "val": "Development", "dev": "Development", "test1": "Test1", "test2": "Test2"}
        dataset_partition = partition_map.get(partition.lower(), partition)
        
        dataset = MSP_Podcast_Dataset(
            audio_root=str(data_dir / "Audios"),
            labels_csv=str(labels_csv),
            transcripts_en_json=str(transcripts_de_json),
            partition=dataset_partition,
            modalities=["text_en"],
        )
        
        # Verificam ca numarul de sample-uri corespunde
        assert len(dataset) == len(embeddings_dict['labels']), \
            "Numarul de sample-uri din datasetul curent nu corespunde cu fisierul existent!"
            
        all_de_embeddings = []
        num_batches = (len(dataset) + self.batch_size - 1) // self.batch_size
        
        # Extragem embeddings
        with torch.no_grad():
            for batch_idx in tqdm(range(num_batches), desc=f"Processing {lang_key} for {partition}"):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, len(dataset))
                
                batch_texts_de = [dataset[idx]["text_en"] for idx in range(start_idx, end_idx)]
                
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
                inputs = tokenizer(batch_texts_de, padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
                outputs = de_backbone(**inputs)
                
                # Extragem hidden states (last hidden state)
                text_de_emb = outputs.last_hidden_state
                
                # Aplicam aceeasi regula de pad/truncate la 100
                text_de_emb = self._pad_or_truncate(text_de_emb, target_len=100).cpu()
                all_de_embeddings.append(text_de_emb)
                
        # Salvam in dictionar
        embeddings_dict[lang_key] = torch.cat(all_de_embeddings, dim=0)
        
        # Actualizare metadata
        embeddings_dict['metadata']['embedding_dims'][lang_key] = embeddings_dict[lang_key].shape[-1]
        embeddings_dict['metadata'][f'extraction_date_{lang_key}'] = datetime.now().isoformat()
        
        print(f"\nSalvare fisier actualizat: {existing_file}")
        torch.save(embeddings_dict, existing_file)
        print(f"[OK] Gata! Forma {lang_key}: {embeddings_dict[lang_key].shape}")


class EmbeddingLoader:
    """Utilitar pentru incarcarea embedding-urilor salvate."""
    
    def __init__(self, embeddings_dir: str = "data/embeddings"):
        self.embeddings_dir = Path(embeddings_dir)
    
    def load_partition(self, partition: str) -> Dict:
        """Incarca embeddings pentru o partitie."""
        embeddings_file = self.embeddings_dir / f"embeddings_{partition}.pt"
        
        if not embeddings_file.exists():
            raise FileNotFoundError(f"Embeddings not found: {embeddings_file}")
        
        embeddings_dict = torch.load(embeddings_file)
        
        print(f"\n[OK] Loaded embeddings for {partition}")
        print(f"  Text EN: {embeddings_dict['text_en'].shape}")
        print(f"  Text ES: {embeddings_dict['text_es'].shape}")
        print(f"  Audio: {embeddings_dict['audio'].shape}")
        print(f"  Labels: {embeddings_dict['labels'].shape}")
        
        return embeddings_dict
    
    def load_all_partitions(self) -> Dict[str, Dict]:
        """Incarca embeddings pentru toate partitiile disponibile."""
        partitions = ['train', 'val', 'test1']
        all_embeddings = {}
        
        for partition in partitions:
            try:
                all_embeddings[partition] = self.load_partition(partition)
            except FileNotFoundError:
                print(f"[!] Skipping {partition} (not found)")
        
        return all_embeddings

def update_embeddings_with_val_arousal(output_dir: str | Path = 'MSP_Podcast/embeddings'):
    """
    Adauga valorile valence (EmoVal) si arousal (EmoAct) din labels_consensus.csv
    in fisierele de embeddings existente, fara a regenera embeddingurile.
    """
    import csv
    import torch
    from pathlib import Path

    def build_label_lookup(labels_csv_path: Path) -> dict[str, tuple[float, float]]:
        lookup: dict[str, tuple[float, float]] = {}
        with labels_csv_path.open('r', encoding='utf-8') as file_handle:
            reader = csv.DictReader(file_handle)
            for row in reader:
                file_name = str(row['FileName'])
                values = (float(row['EmoVal']), float(row['EmoAct']))
                lookup[file_name] = values
                if file_name.endswith('.wav'):
                    lookup[file_name[:-4]] = values
        return lookup

    def add_val_arousal(embeddings_path: Path, label_lookup: dict[str, tuple[float, float]]):
        if not embeddings_path.exists():
            print(f"Skipping missing file: {embeddings_path}")
            return

        print(f"Processing {embeddings_path} ...")
        data = torch.load(embeddings_path, map_location='cpu')
        if 'valence' in data and 'arousal' in data:
            print(f"[OK] {embeddings_path} already contains valence/arousal.")
            return

        file_ids = data['file_ids']
        valence = []
        arousal = []
        for file_id in file_ids:
            key = str(file_id)
            if key not in label_lookup:
                raise KeyError(f"Nu am gasit {file_id} in labels_consensus.csv")
            emo_val, emo_act = label_lookup[key]
            valence.append(emo_val)
            arousal.append(emo_act)

        data['valence'] = torch.tensor(valence, dtype=torch.float32)
        data['arousal'] = torch.tensor(arousal, dtype=torch.float32)
        torch.save(data, embeddings_path)
        print(f"[OK] Updated {embeddings_path} with valence/arousal.")

    base = Path(output_dir)
    labels_csv = Path('MSP_Podcast/Labels/labels_consensus.csv')
    if not labels_csv.exists():
        raise FileNotFoundError(f"Labels CSV not found: {labels_csv}")

    label_lookup = build_label_lookup(labels_csv)
    for partition in ('train', 'val', 'test1'):
        add_val_arousal(base / f'embeddings_{partition}.pt', label_lookup)

def main():
    parser = argparse.ArgumentParser(description="Extract and save embeddings from backbones")
    parser.add_argument(
        '--add-val-arousal',
        action='store_true',
        help='Adauga valence/arousal din labels_consensus.json in embeddings_train.pt si embeddings_val.pt (fara a regenera embeddingurile) si opreste scriptul.'
    )
    parser.add_argument(
        '--partition',
        type=str,
        default='train, val',
        help='Partition to process (default: train, val). Supports comma-separated values.',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for embeddings. Implicit: MSP_Podcast/embeddings pentru text_en,text_es,audio, altfel MSP_Podcast/embeddings_<modalitati>',
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
    parser.add_argument(
        '--add-german-only',
        action='store_true',
        help='Sari peste extragerea de baza si doar adauga embeddings in germana in fisierele existente.',
    )
    parser.add_argument(
        '--en-checkpoint',
        type=str,
        default='checkpoints/roberta_text_en',
        help='Calea catre checkpoint-ul pentru limba engleza',
    )
    parser.add_argument(
        '--es-checkpoint',
        type=str,
        default='checkpoints/roberta_text_es',
        help='Calea catre checkpoint-ul pentru limba spaniola',
    )
    parser.add_argument(
        '--de-checkpoint',
        type=str,
        default='checkpoints/roberta_text_de/best_model',
        help='Calea catre checkpoint-ul pentru limba germana',
    )
    parser.add_argument(
        '--fr-checkpoint',
        type=str,
        default='checkpoints/roberta_text_fr',
        help='Calea catre checkpoint-ul pentru limba franceza',
    )
    parser.add_argument(
        '--audio-checkpoint',
        type=str,
        default='checkpoints/wavlm_audio',
        help='Calea catre checkpoint-ul pentru audio',
    )
    parser.add_argument(
        '--modalities',
        type=str,
        default='text_en,text_es,audio',
        help='Lista de modalitati separate prin virgula. Exemple: text_en,audio sau text_en,text_fr,audio',
    )
    parser.add_argument(
        '--allow-overwrite',
        action='store_true',
        help='Permite suprascrierea fisierelor existente din output-dir.',
    )
    parser.add_argument(
        '--reuse-from',
        type=str,
        default=None,
        help='Lista de directoare de embeddings separate prin ; din care se pot reutiliza modalitatile deja extrase.',
    )
    
    args = parser.parse_args()
    modalities = parse_modalities(args.modalities)
    output_dir = resolve_output_dir(args.output_dir, modalities)
    output_dir_path = Path(output_dir)
    reuse_from_dirs = parse_reuse_dirs(args.reuse_from)

    if (args.add_german_only or args.add_val_arousal) and not args.allow_overwrite:
        raise ValueError(
            "Optiunile --add-german-only si --add-val-arousal modifica fisiere existente. "
            "Ruleaza explicit cu --allow-overwrite daca asta este intentia."
        )

    if not args.add_german_only and not args.add_val_arousal:
        ensure_output_dir_available(output_dir_path, args.allow_overwrite)
    
    # Convert projection_dim
    projection_dim = args.projection_dim if args.projection_dim > 0 else None
    
    print("\n" + "="*80)
    print("EMBEDDING EXTRACTION SCRIPT")
    print("="*80)
    print(f"Partition: {args.partition}")
    print(f"Output dir: {output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Projection dim: {projection_dim}")
    print(f"Device: {args.device}")
    print(f"Modalities: {modalities}")
    print(f"Save individually: {args.save_individually}")
    print(f"Add German Only: {args.add_german_only}")
    print(f"Reuse from dirs: {reuse_from_dirs}")
    print("="*80 + "\n")
    

    # Daca se cere doar adaugarea valence/arousal, ruleaza utilitarul si opreste
    if args.add_val_arousal:
        update_embeddings_with_val_arousal(output_dir)
        print("\nValence/arousal adaugate cu succes in fisierele de embeddings disponibile!\n")
        return

    # Create extractor
    extractor = EmbeddingExtractor(
        output_dir=output_dir,
        text_en_checkpoint=normalize_checkpoint_dir(args.en_checkpoint),
        text_es_checkpoint=normalize_checkpoint_dir(args.es_checkpoint),
        text_de_checkpoint=normalize_checkpoint_dir(args.de_checkpoint),
        text_fr_checkpoint=normalize_checkpoint_dir(args.fr_checkpoint),
        audio_checkpoint=normalize_checkpoint_dir(args.audio_checkpoint),
        projection_dim=projection_dim,
        device=args.device,
        batch_size=args.batch_size,
        modalities=modalities,
        reuse_from_dirs=reuse_from_dirs,
    )

    # Extract embeddings logic
    allowed_partitions = {'train', 'val', 'test1', 'test2', 'dev', 'all'}
    
    if args.add_german_only:
        # adaugare exclusiva embeddings germana
        if args.partition == 'all':
            partitions = ['train', 'val', 'test1']
        else:
            partitions = [p.strip() for p in args.partition.split(',') if p.strip()]
            invalid = [p for p in partitions if p not in allowed_partitions]
            if invalid:
                raise ValueError(
                    f"Invalid partition(s): {', '.join(invalid)}. "
                    f"Allowed: {sorted(allowed_partitions)}"
                )
                
        for partition in partitions:
            extractor.add_german_embeddings(
                partition=partition,
                checkpoint_path=args.de_checkpoint,
                lang_key='text_de'
            )
            
    else:
        # extragere completa (audio + text)
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
    print(f" Embeddings saved to: {output_dir}\n")

if __name__ == "__main__":
    main()


