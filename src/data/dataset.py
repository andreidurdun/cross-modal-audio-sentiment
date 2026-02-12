from __future__ import annotations

import json
import os
from typing import Literal

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.preprocessing.audio_processor import AudioProcessor, AudioProcessorConfig

class MSP_Podcast_Dataset(Dataset):
    """
    Dataset optimizat pentru MSP-Podcast cu clasele: Satisfied, Unsatisfied, Neutral.
    Suporta incarcare selectiva a modulitatatilor: audio, text engleza, text spaniola.
    
    Transcriptiile sunt incarcate din fisiere TXT separate în directoarele specificate.
    
    Utilizare:
        # Doar text engleza din directorul Transcription_en
        dataset = MSP_Podcast_Dataset(
            audio_root='MSP_Podcast/Audios',
            labels_csv='MSP_Podcast/Labels/labels_consensus.csv',
            transcripts_en_dir='MSP_Podcast/Transcription_en',
            modalities=['text_en']
        )
        
        # Doar audio
        dataset = MSP_Podcast_Dataset(
            audio_root='MSP_Podcast/Audios',
            labels_csv='MSP_Podcast/Labels/labels_consensus.csv',
            transcripts_en_dir='MSP_Podcast/Transcription_en',
            modalities=['audio']
        )
        
        # Audio + text engleza
        dataset = MSP_Podcast_Dataset(
            audio_root='MSP_Podcast/Audios',
            labels_csv='MSP_Podcast/Labels/labels_consensus.csv',
            transcripts_en_dir='MSP_Podcast/Transcription_en',
            modalities=['audio', 'text_en']
        )
        
        # Audio + text engleza + text spaniola
        dataset = MSP_Podcast_Dataset(
            audio_root='MSP_Podcast/Audios',
            labels_csv='MSP_Podcast/Labels/labels_consensus.csv',
            transcripts_en_dir='MSP_Podcast/Transcription_en',
            transcripts_es_dir='MSP_Podcast/Transcription_es',  # optional
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
        labels_csv,  # Calea catre 'labels_consensus.csv'
        transcripts_en_dir,  # Director cu fisiere TXT transcriptii engleza (ex: 'MSP_Podcast/Transcription_en')
        transcripts_es_dir=None,  # Optional: director cu fisiere TXT transcriptii spaniola
        partition='Train',  # Ex: 'Train', 'Test1', 'Validation', 'Test2'
        target_sample_rate=16000,
        audio_processor: AudioProcessor | None = None,
        apply_telephony_aug: bool = False,
        modalities: list[Literal['audio', 'text_en', 'text_es']] | None = None,
    ):
        super().__init__()
        
        self.audio_root = audio_root
        self.target_sample_rate = target_sample_rate
        
        # --- Configurare Modalitatile ---
        if modalities is None:
            # Implicit: incarca tot
            modalities = ['audio', 'text_en']
        
        # Validare modalitatile
        invalid_modalities = set(modalities) - set(self.SUPPORTED_MODALITIES)
        if invalid_modalities:
            raise ValueError(
                f"Modalitatile {invalid_modalities} nu sunt suportate. "
                f"Alege dintre: {self.SUPPORTED_MODALITIES}"
            )
        
        self.modalities = modalities
        print(f"Dataset initialized with modalities: {self.modalities}")
        
        # --- Configurare directoare transcripturi ---
        if 'text_en' in self.modalities:
            self.transcripts_en_dir = transcripts_en_dir
            if not os.path.isdir(self.transcripts_en_dir):
                raise ValueError(f"Director invalida: {self.transcripts_en_dir}")
        else:
            self.transcripts_en_dir = None
        
        if 'text_es' in self.modalities:
            if transcripts_es_dir is None:
                raise ValueError(
                    "transcripts_es_dir este necesar cand 'text_es' este in modalities"
                )
            self.transcripts_es_dir = transcripts_es_dir
            if not os.path.isdir(self.transcripts_es_dir):
                raise ValueError(f"Director invalida: {self.transcripts_es_dir}")
        else:
            self.transcripts_es_dir = None
        
        # --- Configurare Audio Processor (doar daca e necesar audio) ---
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
        
        # Filtrare după partiție (Coloana 'Split_Set')
        # Asigură-te că numele partiției corespunde cu cel din CSV (de obicei 'Train', 'Development', 'Test1')
        self.metadata = df[df['Split_Set'] == partition].reset_index(drop=True)
        
        # Aplicăm logica de mapare a claselor direct în DataFrame pentru viteză
        self.metadata['target_label'] = self.metadata.apply(self._map_emotions, axis=1)
        
        # Eliminăm rândurile care nu au putut fi mapate (dacă există)
        self.metadata = self.metadata.dropna(subset=['target_label'])
        
        print(f"Loaded {len(self.metadata)} files for partition: {partition}")

        # Optimizare: Transformator pentru Resampling (reutilizabil)
        self.resample_transform = None 

    def _load_json(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_transcript_from_file(self, transcript_dir: str, file_id: str) -> str:
        """
        Incarca transcription dintr-un fisier TXT individual.
        
        Args:
            transcript_dir: Directorul care contine fisierele TXT
            file_id: ID-ul fisierului (ex: 'MSP-PODCAST_0001_0028')
        
        Returns:
            Continutul transcription, sau string gol daca nu gaseste fisierul
        """
        # Incearca cu si fara extensie .wav
        txt_path = os.path.join(transcript_dir, f"{file_id}.txt")
        
        if not os.path.exists(txt_path):
            # Incearca fara extensie .txt (in caz ca file_id contine deja extensia)
            txt_path = os.path.join(transcript_dir, file_id)
            if not txt_path.endswith('.txt'):
                txt_path += '.txt'
        
        if not os.path.exists(txt_path):
            return ""  # Fisier nu gasit
        
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            return text
        except Exception as e:
            print(f"Error reading transcript {txt_path}: {e}")
            return ""

    def _map_emotions(self, row):
        """
        Logica de conversie: Emoții Originale -> Satisfied/Unsatisfied/Neutral
        Folosește coloanele 'EmoClass' și 'EmoVal'.
        """
        emotion = row['EmoClass']
        valence = float(row['EmoVal'])

        # --- Regula 1: Categorii Negative Clare -> Unsatisfied ---
        if emotion in ['Ang', 'Sad', 'Dis', 'Con', 'Fea']:
            return 'unsatisfied'
        
        # --- Regula 2: Categorie Pozitivă Clară -> Satisfied ---
        elif emotion == 'Hap':
            return 'satisfied'
        
        # --- Regula 3: Neutru -> Neutral ---
        elif emotion == 'Neu':
            return 'neutral'
            
        # --- Regula 4: Dezambiguizare pentru 'Surprise'/'Other' folosind Valența ---
        else:
            if valence <= 3.5:
                return 'unsatisfied'
            elif valence >= 4.5:
                return 'satisfied'
            else:
                return 'neutral'

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # Extragem rândul corespunzător din DataFrame
        row = self.metadata.iloc[idx]
        file_id = row['FileName']
        
        # Inițializare output dict cu intotdeauna label și file_id
        output = {}
        
        # --- 1. Procesare Audio (daca e in modalities) ---
        if 'audio' in self.modalities:
            audio_path = os.path.join(self.audio_root, f"{file_id}")
            if not audio_path.endswith('.wav'): 
                audio_path += '.wav'

            try:
                waveform = self.audio_processor.load_waveform(audio_path)
            except Exception as e:
                print(f"Error loading {file_id}: {e}")
                # Returnăm un tensor gol sau zgomot în caz de eroare pentru a nu opri antrenamentul
                waveform = torch.zeros(self.target_sample_rate)  # 1 secundă de liniște
            
            output['audio'] = waveform  # Shape: (Time,)

        # --- 2. Procesare Text (daca e in modalities) ---
        # Construieste ID-ul fisierului TXT (fara extensie .wav daca exista)
        key_id = file_id.replace('.wav', '')
        
        if 'text_en' in self.modalities:
            text_en = self._load_transcript_from_file(self.transcripts_en_dir, key_id)
            output['text_en'] = text_en
        
        if 'text_es' in self.modalities:
            text_es = self._load_transcript_from_file(self.transcripts_es_dir, key_id)
            output['text_es'] = text_es

        # --- 3. Procesare Label ---
        label_str = row['target_label']
        label_idx = self.LABEL_MAP[label_str]
        
        output['label'] = label_idx  # Int: 0, 1, sau 2
        output['file_id'] = file_id

        return output