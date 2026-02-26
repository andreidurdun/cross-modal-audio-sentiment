from __future__ import annotations

import json
import os
from typing import Literal
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from src.preprocessing.audio_processor import AudioProcessor, AudioProcessorConfig

class MSP_Podcast_Dataset(Dataset):
    """
    Dataset optimizat pentru MSP-Podcast cu clasele: Satisfied, Unsatisfied, Neutral.
    Suporta incarcare selectiva a modulitatatilor: audio, text engleza, text spaniola.
    
    Transcriptiile sunt incarcate din fisiere JSON (cheie -> text).
    
    Utilizare:
        # Doar text engleza din JSON
        dataset = MSP_Podcast_Dataset(
            audio_root='MSP_Podcast/Audios',
            labels_csv='MSP_Podcast/Labels/labels_consensus.csv',
            transcripts_en_json='MSP_Podcast/Transcription_en.json',
            modalities=['text_en']
        )
        
        # Doar audio
        dataset = MSP_Podcast_Dataset(
            audio_root='MSP_Podcast/Audios',
            labels_csv='MSP_Podcast/Labels/labels_consensus.csv',
            transcripts_en_json='MSP_Podcast/Transcription_en.json',
            modalities=['audio']
        )
        
        # Audio + text engleza
        dataset = MSP_Podcast_Dataset(
            audio_root='MSP_Podcast/Audios',
            labels_csv='MSP_Podcast/Labels/labels_consensus.csv',
            transcripts_en_json='MSP_Podcast/Transcription_en.json',
            modalities=['audio', 'text_en']
        )
        
        # Audio + text engleza + text spaniola
        dataset = MSP_Podcast_Dataset(
            audio_root='MSP_Podcast/Audios',
            labels_csv='MSP_Podcast/Labels/labels_consensus.csv',
            transcripts_en_json='MSP_Podcast/Transcription_en.json',
            transcripts_es_json='MSP_Podcast/Transcription_es.json',  # optional
            modalities=['audio', 'text_en', 'text_es']
        )
    """

    # Maparea etichetelor string în indecși numerici
    LABEL_MAP = {
        'unsatisfied': 0,
        'neutral': 1,
        'satisfied': 2
    }
    
    # Modalitatile suportate
    SUPPORTED_MODALITIES = ['audio', 'text_en', 'text_es']

    def __init__(
        self,
        audio_root,
        labels_csv,
        transcripts_en_dir=None,
        transcripts_es_dir=None,
        transcripts_en_json=None,
        transcripts_es_json=None,
        partition='Train',
        target_sample_rate=16000,
        audio_processor: AudioProcessor | None = None,
        apply_telephony_aug: bool = False,
        modalities: list[Literal['audio', 'text_en', 'text_es']] | None = None,
        use_cache: bool = True,
        max_workers: int = 8,
    ):
        super().__init__()
        
        self.audio_root = audio_root
        self.target_sample_rate = target_sample_rate
        self.partition = partition
        
        # --- Configurare Modalitatile ---
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
        
        # --- Configurare directoare transcripturi ---
        self.transcripts_cache_en = {}
        self.transcripts_cache_es = {}
        
        if 'text_en' in self.modalities:
            self.transcripts_en_dir = transcripts_en_dir
            self.transcripts_en_json = transcripts_en_json
            if self.transcripts_en_json is None:
                if not os.path.isdir(self.transcripts_en_dir):
                    raise ValueError(f"Director invalida: {self.transcripts_en_dir}")
        else:
            self.transcripts_en_dir = None
            self.transcripts_en_json = None
        
        if 'text_es' in self.modalities:
            self.transcripts_es_dir = transcripts_es_dir
            self.transcripts_es_json = transcripts_es_json
            if self.transcripts_es_json is None:
                if transcripts_es_dir is None:
                    raise ValueError("transcripts_es_dir este necesar cand 'text_es' este in modalities")
                if not os.path.isdir(self.transcripts_es_dir):
                    raise ValueError(f"Director invalida: {self.transcripts_es_dir}")
        else:
            self.transcripts_es_dir = None
            self.transcripts_es_json = None
        
        # --- Configurare Audio Processor ---
        if 'audio' in self.modalities:
            processor_config = AudioProcessorConfig(
                target_sample_rate=target_sample_rate,
                apply_telephony_augmentation=apply_telephony_aug and partition == 'Train'
            )
            self.audio_processor = audio_processor or AudioProcessor(processor_config)
        else:
            self.audio_processor = None

        # --- 1. Încărcare și Filtrare Metadate (Pandas) ---
        print(f"Loading metadata from {labels_csv}...")
        df = pd.read_csv(labels_csv)
        
        # Filtrare după partiție
        self.metadata = df[df['Split_Set'] == partition].reset_index(drop=True)
        
        # Aplicăm logica de mapare a claselor cu vectorizare
        self.metadata['target_label'] = self._map_emotions_vectorized(self.metadata)
        
        # Eliminăm rândurile care nu au putut fi mapate
        self.metadata = self.metadata.dropna(subset=['target_label'])
        
        # --- OPTIMIZARE: Precompute file IDs și labels ---
        self.metadata['file_id_key'] = self.metadata['FileName'].str.replace('.wav', '', regex=False)
        self.metadata['label_id'] = self.metadata['target_label'].map(self.LABEL_MAP)
        
        print(f"Loaded {len(self.metadata)} files for partition: {partition}")
        
        # --- 2. OPTIMIZARE: Încărcă transcripturile în memorie în PARALEL ---
        if 'text_en' in self.modalities:
            if self.transcripts_en_json is not None:
                print(f"Loading English transcripts from JSON: {self.transcripts_en_json}")
                self.transcripts_cache_en = self._load_transcripts_json(self.transcripts_en_json)
            else:
                print("Preloading English transcripts (PARALLEL)...")
                self.transcripts_cache_en = self._load_transcripts_parallel(
                    self.transcripts_en_dir,
                    max_workers=max_workers,
                    use_cache=use_cache,
                    suffix="_en"
                )
        
        if 'text_es' in self.modalities:
            if self.transcripts_es_json is not None:
                print(f"Loading Spanish transcripts from JSON: {self.transcripts_es_json}")
                self.transcripts_cache_es = self._load_transcripts_json(self.transcripts_es_json)
            else:
                print("Preloading Spanish transcripts (PARALLEL)...")
                self.transcripts_cache_es = self._load_transcripts_parallel(
                    self.transcripts_es_dir,
                    max_workers=max_workers,
                    use_cache=use_cache,
                    suffix="_es"
                )
        
        print(f"Dataset initialization complete! {len(self.metadata)} samples ready\n")

    def _map_emotions_vectorized(self, df: pd.DataFrame) -> pd.Series:
        """Vectorized emotion mapping - MULT mai rapid decât apply()"""
        emotion = df['EmoClass']
        valence = df['EmoVal'].astype(float)
        
        # Condiții vectorizate
        is_negative = emotion.isin(['Ang', 'Sad', 'Dis', 'Con', 'Fea'])
        is_happy = emotion == 'Hap'
        is_neutral = emotion == 'Neu'
        is_low_valence = valence <= 3.5
        is_high_valence = valence >= 4.5
        
        # Selectare vectorizată
        result = pd.Series('neutral', index=df.index, dtype=str)
        result[is_negative] = 'unsatisfied'
        result[is_happy] = 'satisfied'
        result[is_neutral] = 'neutral'
        
        # Pentru altele, folosim valență
        mask_other = ~(is_negative | is_happy | is_neutral)
        result[mask_other & is_low_valence] = 'unsatisfied'
        result[mask_other & is_high_valence] = 'satisfied'
        
        return result
    
    def _load_transcripts_parallel(self, transcript_dir: str, max_workers: int = 8, 
                                    use_cache: bool = True, suffix: str = "") -> dict:
        """
        Încarcă toate transcripturile în PARALEL folosind ThreadPool.
        MULT mai rapid decât citirea secvențială!
        """
        cache_file = Path(transcript_dir).parent / f"{Path(transcript_dir).name}_cache{suffix}.json"
        
        # Check cache first
        if use_cache and cache_file.exists():
            print(f"   Loading from cache: {cache_file}")
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        txt_files = list(Path(transcript_dir).glob("*.txt"))
        print(f"   Found {len(txt_files)} files - Reading in PARALLEL ({max_workers} workers)...")
        
        transcripts = {}
        
        def read_single_file(txt_file):
            """Citește un singur fișier."""
            file_id = txt_file.stem
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    return file_id, f.read().strip()
            except Exception as e:
                return file_id, ""
        
        # Citire paralelă cu ThreadPool
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(read_single_file, f): f for f in txt_files}
            
            for future in tqdm(as_completed(futures), total=len(txt_files), desc="   Reading"):
                file_id, text = future.result()
                transcripts[file_id] = text
        
        # Save cache
        if use_cache:
            print(f"   Saving cache to {cache_file}")
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(transcripts, f, ensure_ascii=False)
        
        print(f"   Cached {len(transcripts)} transcripts")
        return transcripts 

    def _load_transcripts_json(self, transcript_json: str) -> dict:
        """Incarca transcripturile dintr-un JSON (key -> text)."""
        if not os.path.isfile(transcript_json):
            raise ValueError(f"Fisier JSON invalid: {transcript_json}")
        with open(transcript_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("JSON-ul de transcripturi trebuie sa fie un obiect dict")
        print(f"   Loaded {len(data)} transcripts from JSON")
        return data

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        """Get item - OPTIMIZAT cu cache și lazy loading audio"""
        row = self.metadata.iloc[idx]
        file_id = row['FileName']
        file_id_key = row['file_id_key']  # Precomputed, fără '.wav'
        
        output = {}
        
        # --- 1. Audio (Lazy Loading - încarcă doar când cere) ---
        if 'audio' in self.modalities:
            audio_path = os.path.join(self.audio_root, f"{file_id}")
            if not audio_path.endswith('.wav'): 
                audio_path += '.wav'

            try:
                waveform = self.audio_processor.load_waveform(audio_path)
            except Exception as e:
                print(f"EROARE la încărcarea audio {audio_path}: {str(e)}")
                waveform = torch.zeros(self.target_sample_rate)
            
            output['audio'] = waveform
        
        # --- 2. Text EN (din cache preloaded) ---
        if 'text_en' in self.modalities:
            text_en = self.transcripts_cache_en.get(file_id_key, "")
            output['text_en'] = text_en
        
        # --- 3. Text ES (din cache preloaded) ---
        if 'text_es' in self.modalities:
            text_es = self.transcripts_cache_es.get(file_id_key, "")
            output['text_es'] = text_es
        
        # --- 4. Label (precomputed) ---
        output['label'] = int(row['label_id'])
        output['file_id'] = file_id

        return output