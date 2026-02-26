from pathlib import Path
from typing import Optional

import torch
import sys
import numpy as np
from sklearn.metrics import f1_score
import warnings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore", category=UserWarning, module="bitsandbytes")

from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
    BitsAndBytesConfig,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm

# Import dataset optimizat
from src.data.dataset import MSP_Podcast_Dataset


class TextEncoderDataset(Dataset):
    """Dataset wrapper pentru tokenizare text din MSP_Podcast_Dataset."""

    def __init__(self, msp_dataset: MSP_Podcast_Dataset, tokenizer):
        self.msp_dataset = msp_dataset
        self.tokenizer = tokenizer
        # Pre-tokenize everything at init
        self.encodings = self._precompute_encodings()
        self.labels = [int(row['label_id']) for _, row in msp_dataset.metadata.iterrows()]

    def _precompute_encodings(self):
        """Pre-tokenizează toate textele la init (o singură dată)."""
        texts = []
        for idx in range(len(self.msp_dataset)):
            sample = self.msp_dataset[idx]
            text = sample.get('text_en', sample.get('text_es', ''))
            if not text or text.strip() == "":
                text = "[EMPTY]"
            texts.append(text.strip())
        
        print(f"Pre-tokenizing {len(texts)} texts...")
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",  # Forțează o formă perfect dreptunghiulară pentru GPU
            max_length=128,        # Aliniat cu specificul de Twitter al modelului
            return_tensors="pt",   # Garantăm tensori PyTorch
        )
        return encodings

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # adaugă .clone().detach() pentru a preveni warning-uri legate de pointeri în memorie
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()} 
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


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

    def setup_model_with_lora(self, lora_r: int = 16, lora_alpha: int = 32):
        """Configurează modelul cu QLoRA (8-bit + LoRA)."""
        print("Configuring 8-bit Quantization...")
        
        # 8-bit quantization cu skip pentru classifier
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
        )
        
        # Pregătire pentru antrenament pe 8-biți
        model = prepare_model_for_kbit_training(model)
        model.config.use_cache = False

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
        """Antrenează o epocă cu gradient accumulation."""
        model.train()
        total_loss = 0.0

        # Ponderile pentru clase - asigură-te că sunt corecte
        class_weights = torch.tensor([1.02, 1.00, 1.60], dtype=torch.float32).to(self.device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

        progress_bar = tqdm(train_loader, desc="Training", position=0)
        optimizer.zero_grad()
        
        for step, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            
            # Loss ponderat
            loss = loss_fn(outputs.logits, labels)
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Afișăm loss-ul real (neîmpărțit)
            actual_loss = loss.item() * gradient_accumulation_steps
            total_loss += actual_loss
            progress_bar.set_postfix({"loss": f"{actual_loss:.4f}"})

        return total_loss / len(train_loader)

    @torch.no_grad()
    def evaluate(self, model, val_loader: DataLoader) -> tuple[float, float, float]:
        """Evaluează modelul și calculează F1 macro."""
        model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []

        progress_bar = tqdm(val_loader, desc="Evaluating")
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            total_loss += outputs.loss.item()

            # Predictions și labels
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
        learning_rate: float = 2e-4,
        lora_r: int = 16,
        lora_alpha: int = 32,
        gradient_accumulation_steps: int = 1,
    ):
        """Antrenament complet cu QLoRA (8-bit + LoRA)."""
        print("="*80)
        print("Text-Only Training with QLoRA (8-bit)")
        print("="*80)

        # Setup model
        model = self.setup_model_with_lora(lora_r=lora_r, lora_alpha=lora_alpha)

        # Data collator pentru padding dinamic
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        # DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=data_collator,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            collate_fn=data_collator,
        )

        # Optimizer și scheduler
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
        total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps,
        )

        # Training loop
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

            # Train
            train_loss = self.train_epoch(model, train_loader, optimizer, scheduler, gradient_accumulation_steps)
            print(f"Train Loss: {train_loss:.4f}")

            # Evaluate
            val_loss, val_acc, val_f1 = self.evaluate(model, val_loader)
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1 Macro: {val_f1:.4f}")

            # Save best model based on F1 Macro
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_path = checkpoint_dir / "best_model"
                print(f"✅ New best model! F1 Macro: {val_f1:.4f} - Saving to {best_model_path}")
                model.save_pretrained(best_model_path)
                self.tokenizer.save_pretrained(best_model_path)

        print("\n" + "="*80)
        print(f"Training Complete!")
        print(f"Best Validation F1 Macro: {best_val_f1:.4f}")
        print(f"Model saved to: {checkpoint_dir / 'best_model'}")
        print("="*80)


def main():
    """Main training function."""
    
    # Paths
    data_dir = Path("MSP_Podcast")
    labels_csv = data_dir / "Labels" / "labels_consensus.csv"
    transcripts_en_json = data_dir / "Transcription_en.json"
    checkpoint_dir = Path("checkpoints/roberta_text_en")
    
    # Verificare fișiere
    if not labels_csv.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_csv}")
    if not transcripts_en_json.exists():
        raise FileNotFoundError(f"Transcripts JSON not found: {transcripts_en_json}")
    
    print("="*80)
    print("Loading MSP-Podcast English Text Data")
    print("="*80)
    
    # Load train dataset
    print("\n1. Loading Train dataset...")
    train_dataset_msp = MSP_Podcast_Dataset(
        audio_root=str(data_dir / "Audios"),
        labels_csv=str(labels_csv),
        transcripts_en_json=str(transcripts_en_json),
        partition="Train",
        modalities=['text_en'],
        use_cache=True,
        max_workers=8
    )
    
    # Load val dataset
    print("\n2. Loading Validation dataset...")
    val_dataset_msp = MSP_Podcast_Dataset(
        audio_root=str(data_dir / "Audios"),
        labels_csv=str(labels_csv),
        transcripts_en_json=str(transcripts_en_json),
        partition="Development",
        modalities=['text_en'],
        use_cache=True,
        max_workers=8
    )
    
    print(f"\n✅ Data loaded successfully!")
    print(f"   Train: {len(train_dataset_msp)} samples")
    print(f"   Val: {len(val_dataset_msp)} samples\n")
    
    # Create trainer and datasets
    trainer = TextOnlyTrainer()
    train_dataset = TextEncoderDataset(train_dataset_msp, trainer.tokenizer)
    val_dataset = TextEncoderDataset(val_dataset_msp, trainer.tokenizer)
    
    # Training
    trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        checkpoint_dir=checkpoint_dir,
        num_epochs=5,
        batch_size=32,
        learning_rate=2e-4,
        lora_r=16,
        lora_alpha=32,
        gradient_accumulation_steps=1,
    )


if __name__ == "__main__":
    main()
