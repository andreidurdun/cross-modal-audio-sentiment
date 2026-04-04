"""
Script: plot_confusion_matrices.py

- Încarcă modelele individuale LoRA (Text EN, Text ES, Audio) ca și clasificatori
- Încarcă modelul fuzionat (CCMT)
- Rulează predicții pe setul de validare (Development)
- Plotează matricea de confuzie pentru fiecare model în parte
"""

from pathlib import Path
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

# Asigură-te că src/ este în sys.path pentru importuri absolute
try:
    from scripts._bootstrap import project_root
except ModuleNotFoundError:
    from _bootstrap import project_root

PROJECT_ROOT = project_root()

from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, 
    DataCollatorWithPadding, 
    AutoFeatureExtractor,
    AutoModelForSequenceClassification,
    AutoModelForAudioClassification
)
from peft import PeftModel

from src.models.full_model import load_ccmt_only_model  # <-- IMPORT CRITIC PENTRU MODELELE TALE
from src.models import load_full_multimodal_model
from src.data.dataset import MSP_Podcast_Dataset
from src.data.precomputed_embeddings_dataset import PrecomputedEmbeddingsDataset
from src.data.text_datasets import TextEncoderDataset
from src.data.audio_datasets import AudioWaveLMDataset, AudioCollator

# ==========================================
# CONFIGURARE GENERALĂ
# ==========================================
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LABELS = [0, 1, 2]
LABEL_NAMES = ["unsatisfied", "neutral", "satisfied"]
data_dir = PROJECT_ROOT / 'MSP_Podcast'

# ==========================================
# Funcție Helper: Plotează matricea
# ==========================================
def plot_confmat(y_true, y_pred, title, save_path=None):
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()


def main():

    import argparse
    parser = argparse.ArgumentParser(description="Rulează evaluarea pentru modelul dorit.")
    parser.add_argument('--model', type=str, default='all', choices=['all', 'text_en', 'text_es', 'audio', 'ccmt'], help='Model de evaluat')
    args = parser.parse_args()

    # Inițializare DataLoaders (ca înainte)
    tokenizer_en = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    val_dataset_en = TextEncoderDataset(
        MSP_Podcast_Dataset(
            audio_root=str(data_dir / 'Audios'),
            labels_csv=str(data_dir / 'Labels' / 'labels_consensus.csv'),
            transcripts_en_json=str(data_dir / 'Transcription_en.json'),
            partition="Development",
            modalities=['text_en'],
        ),
        tokenizer_en,
        text_fields=["text_en"],
        max_length=128,
        padding="max_length",
    )
    val_loader_en = DataLoader(val_dataset_en, batch_size=BATCH_SIZE, shuffle=False, collate_fn=DataCollatorWithPadding(tokenizer=tokenizer_en))

    tokenizer_es = AutoTokenizer.from_pretrained("pysentimiento/robertuito-sentiment-analysis")
    val_dataset_es = TextEncoderDataset(
        MSP_Podcast_Dataset(
            audio_root=str(data_dir / 'Audios'),
            labels_csv=str(data_dir / 'Labels' / 'labels_consensus.csv'),
            transcripts_es_json=str(data_dir / 'Transcription_es.json'),
            partition="Development",
            modalities=['text_es'],
        ),
        tokenizer_es,
        text_fields=["text_es", "text_en"],
        max_length=128,
        padding="max_length",
    )
    val_loader_es = DataLoader(val_dataset_es, batch_size=BATCH_SIZE, shuffle=False, collate_fn=DataCollatorWithPadding(tokenizer=tokenizer_es))

    feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
    val_dataset_audio = AudioWaveLMDataset(
        MSP_Podcast_Dataset(
            audio_root=str(data_dir / 'Audios'),
            labels_csv=str(data_dir / 'Labels' / 'labels_consensus.csv'),
            partition="Development",
            modalities=['audio'],
        ),
        feature_extractor,
        max_seconds=5,
        do_resample=False,
        label_key="label",
        include_attention_mask=False,
    )
    val_loader_audio = DataLoader(val_dataset_audio, batch_size=BATCH_SIZE, shuffle=False, collate_fn=AudioCollator())

    val_dataset_emb = PrecomputedEmbeddingsDataset(
        embeddings_dir=str(data_dir / "embeddings"),
        partition='val',
        device='cpu',
        regression=False
    )
    val_loader_emb = DataLoader(val_dataset_emb, batch_size=BATCH_SIZE, shuffle=False)

    print("✓ All DataLoaders initialized successfully!\n")



    if args.model in ['all', 'text_en']:
        print("Evaluating Text EN Model...")
        try:
            base_en = AutoModelForSequenceClassification.from_pretrained(
                "cardiffnlp/twitter-roberta-base-sentiment-latest", 
                num_labels=3,
                ignore_mismatched_sizes=True
            )
            text_en_model = PeftModel.from_pretrained(
                base_en, 
                str(PROJECT_ROOT / "checkpoints/roberta_text_en/best_model")
            ).to(DEVICE).eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for batch in val_loader_en:
                    input_ids = batch["input_ids"].to(DEVICE)
                    attention_mask = batch["attention_mask"].to(DEVICE)
                    outputs = text_en_model(input_ids=input_ids, attention_mask=attention_mask)
                    preds = outputs.logits.argmax(dim=-1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(batch['labels'].cpu().numpy())
            plot_confmat(all_labels, all_preds, "Text EN Classifier", "results/cf_matrix/confmat_text_en.pdf")
            print("✓ Text EN evaluated.")
        except Exception as e:
            print(f"⚠ Eroare la evaluarea Text EN: {e}")



    if args.model in ['all', 'text_es']:
        print("\nEvaluating Text ES Model...")
        try:
            base_es = AutoModelForSequenceClassification.from_pretrained(
                "pysentimiento/robertuito-sentiment-analysis", 
                num_labels=3,
                ignore_mismatched_sizes=True
            )
            text_es_model = PeftModel.from_pretrained(
                base_es, 
                str(PROJECT_ROOT / "checkpoints/roberta_text_es/best_model")
            ).to(DEVICE).eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for batch in val_loader_es:
                    input_ids = batch["input_ids"].to(DEVICE)
                    attention_mask = batch["attention_mask"].to(DEVICE)
                    outputs = text_es_model(input_ids=input_ids, attention_mask=attention_mask)
                    preds = outputs.logits.argmax(dim=-1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(batch['labels'].cpu().numpy())
            plot_confmat(all_labels, all_preds, "Text ES Classifier", "results/cf_matrix/confmat_text_es.pdf")
            print("✓ Text ES evaluated.")
        except Exception as e:
            print(f"⚠ Eroare la evaluarea Text ES: {e}")



    if args.model in ['all', 'audio']:
        print("\nEvaluating Audio Model...")
        try:
            base_audio = AutoModelForAudioClassification.from_pretrained(
                "microsoft/wavlm-base-plus", 
                num_labels=3,
                ignore_mismatched_sizes=True
            )
            audio_model = PeftModel.from_pretrained(
                base_audio, 
                str(PROJECT_ROOT / "checkpoints/wavlm_audio/best_model")
            ).to(DEVICE).eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for batch in val_loader_audio:
                    input_values = batch['input_values'].to(DEVICE)
                    attention_mask = batch['attention_mask'].to(DEVICE)
                    outputs = audio_model(input_values=input_values, attention_mask=attention_mask)
                    preds = outputs.logits.argmax(dim=-1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(batch['labels'].cpu().numpy())
            plot_confmat(all_labels, all_preds, "Audio Classifier", "results/cf_matrix/confmat_audio.pdf")
            print("✓ Audio evaluated.")
        except Exception as e:
            print(f"⚠ Eroare la evaluarea Audio: {e}")



    if args.model in ['all', 'ccmt']:
        print("\nEvaluating CCMT Multimodal Model...")
        try:
            # 1. Definim configurația EXACT ca la antrenament
            model_config = {
                'num_classes': 3,
                'text_en_dim': 768,
                'text_es_dim': 768,
                'audio_dim': 768,
                'ccmt_dim': 768,
                'num_patches_per_modality': 100,
                'ccmt_depth': 4,
                'ccmt_heads': 4,
                'ccmt_mlp_dim': 1024,
                'ccmt_dropout': 0.1,
            }
            
            # 2. Creăm modelul gol
            ccmt_model = load_ccmt_only_model(
                text_en_dim=model_config['text_en_dim'],
                text_es_dim=model_config['text_es_dim'],
                audio_dim=model_config['audio_dim'],
                num_classes=model_config['num_classes'],
                ccmt_dim=model_config['ccmt_dim'],
                num_patches_per_modality=model_config['num_patches_per_modality'],
                ccmt_depth=model_config['ccmt_depth'],
                ccmt_heads=model_config['ccmt_heads'],
                ccmt_mlp_dim=model_config['ccmt_mlp_dim'],
                ccmt_dropout=model_config['ccmt_dropout'],
                device=DEVICE,
            )
            
            # 3. ÎNCĂRCĂM GREUTĂȚILE ANTRENATE 
            checkpoint_path = PROJECT_ROOT / "checkpoints/ccmt_multimodal/model2_optim/best_model.pt"
            checkpoint = torch.load(str(checkpoint_path), map_location=DEVICE)
            ccmt_model.load_state_dict(checkpoint['model_state_dict'])
            ccmt_model.eval()
            
            all_preds, all_labels = [], []
            with torch.no_grad():
                for batch in val_loader_emb:
                    text_en_emb = batch['text_en_emb'].to(DEVICE)
                    text_es_emb = batch['text_es_emb'].to(DEVICE)
                    audio_emb = batch['audio_emb'].to(DEVICE)
                    batch_labels = batch.get('labels', batch.get('label'))
                    
                    outputs = ccmt_model(
                        text_en_emb=text_en_emb,
                        text_es_emb=text_es_emb,
                        audio_emb=audio_emb
                    )
                    
                    # Robust extraction of logits tensor
                    if isinstance(outputs, dict):
                        if 'logits' in outputs:
                            logits = outputs['logits']
                        else:
                            # Caută primul tensor din dict
                            logits = next((v for v in outputs.values() if hasattr(v, 'argmax')), None)
                            if logits is None:
                                raise ValueError("Nu s-a găsit tensorul de logits în output-ul modelului CCMT.")
                    else:
                        logits = outputs
                        
                    preds = logits.argmax(dim=-1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(batch_labels.cpu().numpy())
            
            plot_confmat(all_labels, all_preds, "CCMT Multimodal Model", "results/cf_matrix/confmat_ccmt.pdf")
            print("✓ CCMT evaluated.")
        except Exception as e:
            print(f"⚠ Eroare la evaluarea CCMT: {e}")

    print("\n" + "="*50)
    print("✓ Toate matricile de confuzie au fost salvate cu succes în 'results/cf_matrix/'!")
    print("="*50)


if __name__ == "__main__":
    main()