"""Custom collate functions for MSP-Podcast datasets."""
import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    """
    Collate function for MSP-Podcast dataset cu suport pentru modalitatile selective.
    
    Supported modalities:
    - 'audio': Audio waveform
    - 'text_en': English transcription
    - 'text_es': Spanish transcription
    
    Functia detectează automat ce modalitatile sunt prezente în batch.
    """
    
    # Detectează ce modalitatile sunt prezente
    has_audio = 'audio' in batch[0]
    has_text_en = 'text_en' in batch[0]
    has_text_es = 'text_es' in batch[0]
    
    output = {}
    
    # --- Procesare Audio (daca exista) ---
    if has_audio:
        audio_list = [item['audio'] for item in batch]
        audio_padded = pad_sequence(audio_list, batch_first=True, padding_value=0.0)
        
        # Creeaza attention mask
        attention_mask = torch.zeros_like(audio_padded)
        for idx, audio in enumerate(audio_list):
            attention_mask[idx, : len(audio)] = 1
        
        output['audio'] = audio_padded
        output['attention_mask'] = attention_mask
    
    # --- Procesare Text Engleza (daca exista) ---
    if has_text_en:
        text_en = [item['text_en'] for item in batch]
        output['text_en'] = text_en
    
    # --- Procesare Text Spaniola (daca exista) ---
    if has_text_es:
        text_es = [item['text_es'] for item in batch]
        output['text_es'] = text_es
    
    # --- Procesare Labels (intotdeauna prezent) ---
    labels = torch.LongTensor([item['label'] for item in batch])
    output['labels'] = labels
    
    # --- Procesare File IDs (intotdeauna prezent) ---
    file_ids = [item['file_id'] for item in batch]
    output['file_ids'] = file_ids
    
    return output


def collate_fn_text_only(batch):
    """
    Collate function optimizat pentru text-only datasets.
    Mai eficient pentru antrenarea pe text fara audio.
    """
    output = {
        'text': [item.get('text_en') or item.get('text_es') or item.get('text') for item in batch],
        'labels': torch.LongTensor([item['label'] for item in batch]),
        'file_ids': [item['file_id'] for item in batch],
    }
    return output


def collate_fn_audio_only(batch):
    """
    Collate function optimizat pentru audio-only datasets.
    Mai eficient pentru antrenarea pe audio fara text.
    """
    audio_list = [item['audio'] for item in batch]
    audio_padded = pad_sequence(audio_list, batch_first=True, padding_value=0.0)
    
    # Creeaza attention mask
    attention_mask = torch.zeros_like(audio_padded)
    for idx, audio in enumerate(audio_list):
        attention_mask[idx, : len(audio)] = 1
    
    output = {
        'audio': audio_padded,
        'attention_mask': attention_mask,
        'labels': torch.LongTensor([item['label'] for item in batch]),
        'file_ids': [item['file_id'] for item in batch],
    }
    return output
