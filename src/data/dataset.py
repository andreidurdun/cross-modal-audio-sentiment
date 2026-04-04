from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.audio_processor import AudioProcessor, AudioProcessorConfig

class MSP_Podcast_Dataset(Dataset):
 
    LABEL_MAP = {
        'unsatisfied': 0,
        'neutral': 1,
        'satisfied': 2
    }
    

    SUPPORTED_MODALITIES = ['audio', 'text_en', 'text_es', 'text_de', 'text_fr']

    def __init__(
        self,
        audio_root,
        labels_csv,
        transcripts_en_json=None,
        transcripts_es_json=None,
        transcripts_de_json=None,
        transcripts_fr_json=None,
        partition='Train',
        target_sample_rate=16000,
        audio_processor: AudioProcessor | None = None,
        modalities: list[str] | None = None,
    ):
        super().__init__()
        
        self.audio_root = audio_root
        self.target_sample_rate = target_sample_rate
        self.partition = partition
        
    
        if modalities is None:
            modalities = ['audio', 'text_en']
        
        invalid_modalities = set(modalities) - set(self.SUPPORTED_MODALITIES)
        if invalid_modalities:
            raise ValueError(
                f"Modalitatile {invalid_modalities} nu sunt suportate. "
                f"Alege dintre: {self.SUPPORTED_MODALITIES}"
            )
        
        self.modalities = modalities
        print(f"Dataset initialized with modalities: {self.modalities}")
        
        # configurare directoare transcripturi
        self.transcript_json_paths = {
            'text_en': transcripts_en_json,
            'text_es': transcripts_es_json,
            'text_de': transcripts_de_json,
            'text_fr': transcripts_fr_json,
        }
        self.transcript_caches = {
            'text_en': {},
            'text_es': {},
            'text_de': {},
            'text_fr': {},
        }

        for modality in self.modalities:
            if not modality.startswith('text_'):
                continue
            transcript_path = self.transcript_json_paths.get(modality)
            if transcript_path is None:
                raise ValueError(f"Fisier invalid pentru {modality}: {transcript_path}")
        
        #configurare Audio Processor
        if 'audio' in self.modalities:
            processor_config = AudioProcessorConfig(
                target_sample_rate=target_sample_rate
            )
            self.audio_processor = audio_processor or AudioProcessor(processor_config)
        else:
            self.audio_processor = None

        #incarcare si filtrare metadate Pandas
        print(f"Loading metadata from {labels_csv}...")
        df = pd.read_csv(labels_csv)
        
        #filtram dupa partitie si resetam indexul pentru acces rapid   
        self.metadata = df[df['Split_Set'] == partition].reset_index(drop=True)
        
        #maparea claselor de emotii la cele 3 clase finale
        self.metadata['target_label'] = self._map_emotions_vectorized(self.metadata)
        
        #eliminam randurile fara label valid
        self.metadata = self.metadata.dropna(subset=['target_label'])
        
        #optimizare: precompute file IDs si labels
        self.metadata['file_id_key'] = self.metadata['FileName'].str.replace('.wav', '', regex=False)
        self.metadata['label_id'] = self.metadata['target_label'].map(self.LABEL_MAP)
        
        print(f"Loaded {len(self.metadata)} files for partition: {partition}")
     
        #incarare transcripturi in memorie
        language_labels = {
            'text_en': 'English',
            'text_es': 'Spanish',
            'text_de': 'German',
            'text_fr': 'French',
        }
        for modality in self.modalities:
            if not modality.startswith('text_'):
                continue
            transcript_path = self.transcript_json_paths[modality]
            print(f"Loading {language_labels.get(modality, modality)} transcripts from JSON: {transcript_path}")
            self.transcript_caches[modality] = self._load_transcripts_json(transcript_path)
                
        print(f"Dataset initialization complete! {len(self.metadata)} samples ready\n")

    def _map_emotions_vectorized(self, df: pd.DataFrame) -> pd.Series:
        emotion = df['EmoClass']
        valence = df['EmoVal'].astype(float)
        
        is_negative = emotion.isin(['Ang', 'Sad', 'Dis', 'Con', 'Fea'])
        is_happy = emotion == 'Hap'
        is_neutral = emotion == 'Neu'
        is_low_valence = valence <= 3.5
        is_high_valence = valence >= 4.5
        
      
        
        result = pd.Series('neutral', index=df.index, dtype=str)
        result[is_negative] = 'unsatisfied'
        result[is_happy] = 'satisfied'
        result[is_neutral] = 'neutral'
        

        mask_other = ~(is_negative | is_happy | is_neutral)
        result[mask_other & is_low_valence] = 'unsatisfied'
        result[mask_other & is_high_valence] = 'satisfied'
        
        return result
    
    
    def _load_transcripts_json(self, transcript_json: str) -> dict:
      
        if not os.path.isfile(transcript_json):
            raise ValueError(f"Fisier JSON invalid: {transcript_json}")
        with open(transcript_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("JSON-ul de transcripturi trebuie sa fie un obiect dict")
        print(f"Loaded {len(data)} transcripts from JSON")
        return data

    def get_class_weights(self, all_train_labels, device):
 
        classes = np.unique(all_train_labels)
        
        weights = compute_class_weight(
            class_weight='balanced', 
            classes=classes, 
            y=all_train_labels
        )
        
        class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
        
        return class_weights_tensor

    def get_max_text_length(self):
        """Returneaza lungimea celui mai lung text din dataset"""
        max_length = 0
        
        for idx in range(len(self)):
            sample = self[idx]
            
            for modality in self.modalities:
                if not modality.startswith('text_'):
                    continue
                text_value = sample.get(modality, '')
                if text_value:
                    max_length = max(max_length, len(text_value))
        
        return max_length

    def get_max_audio_length_seconds(self):
        """Returneaza durata maxima in secunde a unei inregistrari audio din dataset"""
        if 'audio' not in self.modalities or self.audio_processor is None:
            raise ValueError("Modalitatea 'audio' nu este activata in dataset")
        
        max_duration_seconds = 0.0
        
        for idx in tqdm(range(len(self.metadata)), desc="Calculating max audio duration"):
            row = self.metadata.iloc[idx]
            file_id = row['FileName']
            audio_path = os.path.join(self.audio_root, f"{file_id}")
            
            if not audio_path.endswith('.wav'):
                audio_path += '.wav'
            
            try:
                waveform = self.audio_processor.load_waveform(audio_path)
                duration_seconds = len(waveform) / self.target_sample_rate
                max_duration_seconds = max(max_duration_seconds, duration_seconds)
            except Exception as e:
                print(f"[WARN] Eroare la incarcarea audio {audio_path}: {str(e)}")
                continue
        
        return max_duration_seconds

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):

        row = self.metadata.iloc[idx]
        file_id = row['FileName']
        file_id_key = row['file_id_key']  #precomputed, fara .wav
        
        output = {}
        
        #lazy loading audio
        if 'audio' in self.modalities:
            audio_path = os.path.join(self.audio_root, f"{file_id}")
            if not audio_path.endswith('.wav'): 
                audio_path += '.wav'

            try:
                waveform = self.audio_processor.load_waveform(audio_path)
            except Exception as e:
                print(f"EROARE la incarcarea audio {audio_path}: {str(e)}")
                waveform = torch.zeros(self.target_sample_rate)
            
            output['audio'] = waveform
        
        #text en din cache
        for modality in self.modalities:
            if not modality.startswith('text_'):
                continue
            output[modality] = self.transcript_caches[modality].get(file_id_key, "")
        
        #labelul final
        output['label'] = int(row['label_id'])
        output['label_id'] = int(row['label_id'])
        output['valence'] = float(row['EmoVal'])
        output['arousal'] = float(row['EmoAct'])
        output['val_arousal'] = torch.tensor([float(row['EmoVal']), float(row['EmoAct'])], dtype=torch.float32)
        output['file_id'] = file_id

        return output


def main():
    data_dir = PROJECT_ROOT / 'MSP_Podcast'

    train_dataset_msp = MSP_Podcast_Dataset(
        audio_root=str(data_dir / 'Audios'),
        labels_csv=str(data_dir / 'Labels' / 'labels_consensus.csv'),
        transcripts_en_json=str(data_dir / 'Transcription_en.json'),
        partition="Train",
        modalities=['text_en'],
    )

    val_dataset_msp = MSP_Podcast_Dataset(
        audio_root=str(data_dir / 'Audios'),
        labels_csv=str(data_dir / 'Labels' / 'labels_consensus.csv'),
        transcripts_en_json=str(data_dir / 'Transcription_en.json'),
        partition="Development",
        modalities=['text_en'],
    )


    # Numar total exemple
    print(f"\nNumar total exemple (Train): {len(train_dataset_msp)}")
    print(f"Numar total exemple (Val): {len(val_dataset_msp)}")

    # Numar exemple per clasa (Train)
    train_label_counts = (
        train_dataset_msp.metadata['target_label']
        .value_counts()
        .reindex(['unsatisfied', 'neutral', 'satisfied'], fill_value=0)
    )
    print("\nNumar exemple per label (Train):")
    for label_name, count in train_label_counts.items():
        print(f"  - {label_name}: {count}")

    # Numar exemple per clasa (Val)
    val_label_counts = (
        val_dataset_msp.metadata['target_label']
        .value_counts()
        .reindex(['unsatisfied', 'neutral', 'satisfied'], fill_value=0)
    )
    print("\nNumar exemple per label (Val):")
    for label_name, count in val_label_counts.items():
        print(f"  - {label_name}: {count}")

    class_weights = train_dataset_msp.get_class_weights(
        all_train_labels=train_dataset_msp.metadata['label_id'].values,
        device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    print(f"Class weights: {class_weights}")

    max_length = train_dataset_msp.get_max_text_length()
    print(f"Lungimea maxima a textului: {max_length}")

if __name__ == "__main__":
    main()

    