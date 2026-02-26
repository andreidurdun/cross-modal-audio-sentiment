"""
Antrenament RoBERTuito cu QLoRA (8-bit) folosind text spaniolă din traducerile ES.
Script optimizat pentru acuratețe maximă și consum redus de memorie (RTX 4060).
Model: pysentimiento/robertuito-sentiment-analysis
"""
from pathlib import Path
from typing import Optional
import sys
import warnings
import numpy as np
import json

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig, 
    TaskType, 
    get_peft_model, 
    prepare_model_for_kbit_training
)
from tqdm import tqdm
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Suppress bitsandbytes quantization warnings
warnings.filterwarnings("ignore", category=UserWarning, module="bitsandbytes")

from src.data.dataset import MSP_Podcast_Dataset


class TextEncoderDataset(Dataset):
    """Dataset wrapper pentru tokenizare text din MSP_Podcast_Dataset."""

    def __init__(self, msp_dataset: MSP_Podcast_Dataset, tokenizer):
        self.msp_dataset = msp_dataset
        self.tokenizer = tokenizer
        
        # Validare sigură a etichetelor (Prevenim CUDA device-side assert error)
        self.labels = []
        for _, row in msp_dataset.metadata.iterrows():
            lbl = int(row['label_id'])
            if lbl not in [0, 1, 2]:
                raise ValueError(f"CRITICAL: Found invalid label '{lbl}'. Must be 0, 1, or 2.")
            self.labels.append(lbl)
            
        # Pre-tokenize everything at init
        self.encodings = self._precompute_encodings()

    def _precompute_encodings(self):
        """Pre-tokenizează toate textele la init (o singură dată)."""
        texts = []
        for idx in range(len(self.msp_dataset)):
            sample = self.msp_dataset[idx]
            text = sample.get('text_es', sample.get('text_en', ''))
            if not text or text.strip() == "":
                text = "[EMPTY]"
            texts.append(text.strip())
        
        print(f"Pre-tokenizing {len(texts)} texts...")
        # REPARAT: padding='max_length', return_tensors='pt', max_length=128
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",  
            max_length=128,        # Robertuito e antrenat pe tweet-uri, 128 este ideal!
            return_tensors="pt",   # Garantăm că ieșirea e formată din tensori egali
        )
        return encodings

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Extragem feliile de tensor pentru sample-ul curent
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class TextOnlyTrainerES:
    """Trainer pentru antrenament text-only cu QLoRA pentru spaniolă."""

    def __init__(
        self,
        model_name: str = "pysentimiento/robertuito-sentiment-analysis",
        num_labels: int = 3,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.num_labels = num_labels

        self.id2label = {0: "unsatisfied", 1: "neutral", 2: "satisfied"}
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def setup_model_with_lora(self, lora_r: int = 16, lora_alpha: int = 32):
        print("Configuring 8-bit Quantization...")
        
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_skip_modules=["classifier"]
        )

        print("Loading base model in 8-bit...")
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True,
            quantization_config=bnb_config,
            # Fallback opțional dacă mai iei erori cu SDPA pe PyTorch 2.0+
            # attn_implementation="eager" 
        )

        model = prepare_model_for_kbit_training(model)
        model.config.use_cache = False 
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        print("Configuring LoRA...")
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.1,
            bias="none",
            target_modules=["query", "value", "key", "output.dense", "intermediate.dense"], 
            modules_to_save=["classifier"], 
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        return model.to(self.device)

    def train_epoch(
        self,
        model,
        train_loader: DataLoader,
        optimizer,
        scheduler,
        gradient_accumulation_steps: int = 1,
    ) -> float:
        model.train()
        total_loss = 0.0

        # Ponderile pentru clase
        class_weights = torch.tensor([1.02, 1.00, 1.60], dtype=torch.float32).to(self.device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

        progress_bar = tqdm(train_loader, desc="Training", position=0)
        optimizer.zero_grad(set_to_none=True) # Optimizare extra pt PyTorch
        
        for step, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = loss_fn(outputs.logits, labels)
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            actual_loss = loss.item() * gradient_accumulation_steps
            total_loss += actual_loss
            progress_bar.set_postfix({"loss": f"{actual_loss:.4f}"})

        return total_loss / len(train_loader)

    @torch.no_grad()
    def evaluate(self, model, val_loader: DataLoader) -> tuple[float, float, float]:
        model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []

        progress_bar = tqdm(val_loader, desc="Evaluating", position=0, leave=False)
        
        # Ponderile pt loss de validare
        class_weights = torch.tensor([1.02, 1.00, 1.60], dtype=torch.float32).to(self.device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

        for batch in progress_bar:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            loss = loss_fn(outputs.logits, labels)
            total_loss += loss.item()

            predictions = outputs.logits.argmax(dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
        f1_macro = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
        return avg_loss, accuracy, f1_macro

    def train(
        self,
        train_dataset,
        val_dataset,
        checkpoint_dir: Path,
        num_epochs: int = 5,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        lora_r: int = 16,
        lora_alpha: int = 32,
        gradient_accumulation_steps: int = 1,
    ):
        print("="*80)
        print("Spanish Text-Only Training with QLoRA (8-bit)")
        print("="*80)

        model = self.setup_model_with_lora(lora_r=lora_r, lora_alpha=lora_alpha)

        # Setat num_workers=0 pentru citire in-memory
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=0)

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps,
        )

        best_val_f1 = 0.0
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Batch size: {batch_size} (Effective: {batch_size * gradient_accumulation_steps})")
        print(f"Learning rate: {learning_rate}")
        print(f"LoRA r={lora_r}, alpha={lora_alpha}")
        print(f"Best model saved by: F1 Macro\n")

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 80)

            train_loss = self.train_epoch(model, train_loader, optimizer, scheduler, gradient_accumulation_steps)
            print(f"Train Loss: {train_loss:.4f}")

            val_loss, val_acc, val_f1 = self.evaluate(model, val_loader)
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1 Macro: {val_f1:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_path = checkpoint_dir / "best_model"
                print(f"New best model! F1 Macro: {val_f1:.4f} - Saving to {best_model_path}")
                model.save_pretrained(best_model_path)
                self.tokenizer.save_pretrained(best_model_path)

        print("\n" + "="*80)
        print(f"Training Complete!")
        print(f"Best Validation F1 Macro: {best_val_f1:.4f}")
        print("="*80)


def main():
    data_dir = Path("MSP_Podcast")
    labels_csv = data_dir / "Labels" / "labels_consensus.csv"
    transcripts_es_json = data_dir / "Transcription_es.json"
    checkpoint_dir = Path("checkpoints/roberta_text_es")
    
    if not labels_csv.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_csv}")
    if not transcripts_es_json.exists():
        raise FileNotFoundError(f"Transcripts JSON not found: {transcripts_es_json}")
    
    print("="*80)
    print("Loading MSP-Podcast Spanish Text Data")
    print("="*80)

    print("\n1. Loading Train dataset...")
    train_dataset_msp = MSP_Podcast_Dataset(
        audio_root=str(data_dir / "Audios"),
        labels_csv=str(labels_csv),
        transcripts_es_json=str(transcripts_es_json),
        partition="Train",
        modalities=['text_es'],
        use_cache=True,
        max_workers=8
    )
    
    print("\n2. Loading Validation dataset...")
    val_dataset_msp = MSP_Podcast_Dataset(
        audio_root=str(data_dir / "Audios"),
        labels_csv=str(labels_csv),
        transcripts_es_json=str(transcripts_es_json),
        partition="Development",
        modalities=['text_es'],
        use_cache=True,
        max_workers=8
    )

    print(f"\n✅ Data loaded successfully!")
    print(f"   Train: {len(train_dataset_msp)} samples")
    print(f"   Val: {len(val_dataset_msp)} samples\n")
    
    trainer = TextOnlyTrainerES()
    train_dataset = TextEncoderDataset(train_dataset_msp, trainer.tokenizer)
    val_dataset = TextEncoderDataset(val_dataset_msp, trainer.tokenizer)
    
    trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        checkpoint_dir=checkpoint_dir,
        num_epochs=5,
        batch_size=64, # Perfect pt VRAM cu max_length=128
        learning_rate=2e-4, 
        lora_r=16,
        lora_alpha=32,
        gradient_accumulation_steps=1,
    )


if __name__ == "__main__":
    main()