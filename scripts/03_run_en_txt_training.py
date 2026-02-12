"""
Antrenament RoBERTa cu LoRA folosind DOAR text engleza din transcriptiile TXT.
Exemplu complet de utilizare a dataset-ului cu modalitatile selective.
"""
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from src.data.collate import collate_fn_text_only
from src.data.dataset import MSP_Podcast_Dataset
from torch.utils.data import DataLoader, random_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm


class TextOnlyTrainer:
    """Trainer pentru antrenament text-only cu LoRA."""

    def __init__(
        self,
        model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        num_labels: int = 3,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.num_labels = num_labels

        # Label mapping
        self.id2label = {0: "unsatisfied", 1: "neutral", 2: "satisfied"}
        self.label2id = {v: k for k, v in self.id2label.items()}

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def setup_model_with_lora(self, lora_r: int = 8, lora_alpha: int = 16):
        """Configurează modelul cu LoRA."""
        print("Loading base model...")
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
        )

        print("Configuring LoRA...")
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            bias="none",
            target_modules=["q_proj", "v_proj"],  # RoBERTa attention modules
            modules_to_save=["classifier"],  # Save classification head
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        return model.to(self.device)

    def tokenize_texts(self, texts: list[str], max_length: int = 512) -> dict:
        """Tokenizează texte în batch."""
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        return encodings

    def train_epoch(
        self,
        model,
        train_loader: DataLoader,
        optimizer,
        scheduler,
    ) -> float:
        """Antrenează o epocă."""
        model.train()
        total_loss = 0.0

        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            # Tokenizează textele
            texts = batch["text"]
            encodings = self.tokenize_texts(texts)

            input_ids = encodings["input_ids"].to(self.device)
            attention_mask = encodings["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        return total_loss / len(train_loader)

    def evaluate(self, model, eval_loader: DataLoader) -> tuple[float, float]:
        """Evaluează modelul."""
        model.eval()
        total_loss = 0.0
        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                # Tokenizează textele
                texts = batch["text"]
                encodings = self.tokenize_texts(texts)

                input_ids = encodings["input_ids"].to(self.device)
                attention_mask = encodings["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                total_loss += loss.item()

                # Predictions
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(eval_loader)
        accuracy = np.mean(np.array(predictions) == np.array(true_labels))

        return avg_loss, accuracy


def main():
    print("=" * 80)
    print("RoBERTa Text-Only Training with LoRA")
    print("=" * 80)

    # --- Configurare Paths ---
    audio_root = Path("MSP_Podcast/Audios")
    labels_csv = Path("MSP_Podcast/Labels/labels_consensus.csv")
    transcripts_en_dir = Path("MSP_Podcast/Transcription_en")

    # --- Încărcare Dataset (DOAR TEXT ENGLEZA) ---
    print("\n1. Loading dataset with modalities=['text_en']...")
    train_dataset = MSP_Podcast_Dataset(
        audio_root=audio_root,
        labels_csv=labels_csv,
        transcripts_en_dir=transcripts_en_dir,
        partition="Train",
        modalities=["text_en"],  #DOAR TEXT ENGLEZA
    )

    val_dataset = MSP_Podcast_Dataset(
        audio_root=audio_root,
        labels_csv=labels_csv,
        transcripts_en_dir=transcripts_en_dir,
        partition="Validation",
        modalities=["text_en"], 
    )

    
    print(f"   Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # --- DataLoaders ---
    print("\n2. Creating DataLoaders with collate_fn_text_only...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_fn_text_only,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        collate_fn=collate_fn_text_only,
    )

    # --- Setup Trainer și Model ---
    print("\n3. Setting up trainer and model with LoRA...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = TextOnlyTrainer(device=device)
    model = trainer.setup_model_with_lora(lora_r=8, lora_alpha=16)

    # --- Optimizer și Scheduler ---
    num_epochs = 5
    num_training_steps = len(train_loader) * num_epochs

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=500,
        num_training_steps=num_training_steps,
    )

    # --- Antrenament ---
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80 + "\n")

    best_val_accuracy = 0.0
    checkpoint_dir = Path("checkpoints/roberta_text_en")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 40)

        # Train
        train_loss = trainer.train_epoch(model, train_loader, optimizer, scheduler)
        print(f"Train Loss: {train_loss:.4f}")

        # Evaluate
        val_loss, val_accuracy = trainer.evaluate(model, val_loader)
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {val_accuracy:.4f}")

        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_checkpoint = checkpoint_dir / "best_model"
            best_checkpoint.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(best_checkpoint))
            trainer.tokenizer.save_pretrained(str(best_checkpoint))
            print(f"✓ Best model saved: {best_checkpoint}")

    print("\n" + "=" * 80)
    print(" Training complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
