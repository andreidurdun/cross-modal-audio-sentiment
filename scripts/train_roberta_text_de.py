from pathlib import Path
from typing import Optional
import sys
import warnings
import numpy as np
import json
import time
import matplotlib.pyplot as plt

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

warnings.filterwarnings("ignore", category=UserWarning, module="bitsandbytes")
warnings.filterwarnings("ignore", message=".*use_reentrant parameter should be passed explicitly.*")

from src.data.dataset import MSP_Podcast_Dataset

class TextEncoderDataset(Dataset):
    def __init__(self, msp_dataset: MSP_Podcast_Dataset, tokenizer):
        self.msp_dataset = msp_dataset
        self.tokenizer = tokenizer
        self.labels = []
        for _, row in msp_dataset.metadata.iterrows():
            lbl = int(row['label_id'])
            if lbl not in [0, 1, 2]:
                raise ValueError(f"CRITICAL: Found invalid label '{lbl}'. Must be 0, 1, or 2.")
            self.labels.append(lbl)
        self.encodings = self._precompute_encodings()

    def _precompute_encodings(self):
        texts = []
        for idx in range(len(self.msp_dataset)):
            sample = self.msp_dataset[idx]
            text = sample.get('text_de', sample.get('text_en', ''))
            if not text or text.strip() == "":
                text = "[EMPTY]"
            texts.append(text.strip())
        print(f"Pre-tokenizing {len(texts)} texts...")
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )
        return encodings

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

class TextOnlyTrainerDE:
    def __init__(
        self,
        model_name: str = "deepset/gbert-base",
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
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            llm_int8_skip_modules=["classifier"]
        )
        print("Loading base model in 4 bit...")
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True,
            quantization_config=bnb_config,
        )
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
        class_weights: Optional[torch.Tensor] = None,
    ) -> float:
        model.train()
        total_loss = 0.0
        if class_weights is not None:
            loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
        else:
            loss_fn = torch.nn.CrossEntropyLoss()
        progress_bar = tqdm(
            train_loader,
            desc="Training",
            position=0,
            leave=True,
            dynamic_ncols=True,
            mininterval=0.5,
        )
        optimizer.zero_grad()
        for step, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
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
    def evaluate(
        self,
        model,
        val_loader: DataLoader,
        class_weights: Optional[torch.Tensor] = None,
    ) -> tuple[float, float, float]:
        model.eval()
        total_loss = 0.0
        total_samples = 0
        all_predictions = []
        all_labels = []
        if class_weights is not None:
            loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
        else:
            loss_fn = torch.nn.CrossEntropyLoss()
        progress_bar = tqdm(val_loader, desc="Evaluating", position=0, leave=False)
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            loss = loss_fn(outputs.logits, labels)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            predictions = outputs.logits.argmax(dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        avg_loss = total_loss / total_samples
        accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
        f1_macro = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
        return float(avg_loss), float(accuracy), float(f1_macro)

    def _plot_training_curves(self, train_losses, val_losses, checkpoint_dir):
        epochs = range(1, len(train_losses) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2, marker='o')
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, marker='s')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = checkpoint_dir / "training_curves.pdf"
        plt.savefig(plot_path, format='pdf', bbox_inches='tight')
        plt.close()
        print(f"Training curves saved to: {plot_path}")

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
        class_weights: Optional[torch.Tensor] = None,
    ):
        start_time = time.time()
        model = self.setup_model_with_lora(lora_r=lora_r, lora_alpha=lora_alpha)
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
        best_val_loss = float('inf')
        best_val_acc = 0.0
        last_train_loss = 0.0
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        train_losses = []
        val_losses = []
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Batch size: {batch_size} (Effective: {batch_size * gradient_accumulation_steps})")
        print(f"Learning rate: {learning_rate}")
        print(f"LoRA r={lora_r}, alpha={lora_alpha}")
        print(f"Best model saved by: F1 Macro\n")
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 80)
            train_loss = self.train_epoch(model, train_loader, optimizer, scheduler, gradient_accumulation_steps, class_weights)
            train_losses.append(train_loss)
            last_train_loss = train_loss
            print(f"Train Loss: {train_loss:.4f}")
            val_loss, val_acc, val_f1 = self.evaluate(model, val_loader, class_weights)
            val_losses.append(val_loss)
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1 Macro: {val_f1:.4f}")
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_model_path = checkpoint_dir / "best_model"
                print(f"New best model! F1 Macro: {val_f1:.4f} - Saving to {best_model_path}")
                model.save_pretrained(str(best_model_path))
                self.tokenizer.save_pretrained(str(best_model_path))
        end_time = time.time()
        total_time_seconds = end_time - start_time
        total_time_minutes = total_time_seconds / 60
        total_time_hours = total_time_minutes / 60
        print("\n" + "="*80)
        print(f"Training Complete!")
        print(f"Best Validation F1 Macro: {best_val_f1:.4f}")
        print(f"Model saved to: {checkpoint_dir / 'best_model'}")
        print("="*80)
        training_results = {
            "total_training_time": {
                "seconds": total_time_seconds,
                "minutes": total_time_minutes,
                "hours": total_time_hours
            },
            "best_model_metrics": {
                "val_loss": float(best_val_loss),
                "val_accuracy": float(best_val_acc),
                "val_f1_macro": float(best_val_f1)
            },
            "final_train_metrics": {
                "train_loss": float(last_train_loss)
            },
            "hyperparameters": {
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "gradient_accumulation_steps": gradient_accumulation_steps
            },
            "training_curves": {
                "train_losses": [float(l) for l in train_losses],
                "val_losses": [float(l) for l in val_losses]
            }
        }
        results_file = checkpoint_dir / "training_results.json"
        with open(results_file, 'w') as f:
            json.dump(training_results, f, indent=2)
        print(f"\nTraining results saved to: {results_file}")
        self._plot_training_curves(train_losses, val_losses, checkpoint_dir)

def main():
    data_dir = Path("MSP_Podcast")
    labels_csv = data_dir / "Labels" / "labels_consensus.csv"
    transcripts_de_json = data_dir / "Transcription_de.json"
    checkpoint_dir = Path("checkpoints/roberta_text_de")
    if not labels_csv.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_csv}")
    if not transcripts_de_json.exists():
        raise FileNotFoundError(f"Transcripts JSON not found: {transcripts_de_json}")
    print("="*80)
    print("Loading MSP-Podcast German Text Data")
    print("="*80)
    print("\n1. Loading Train dataset...")
    train_dataset_msp = MSP_Podcast_Dataset(
        audio_root=str(data_dir / "Audios"),
        labels_csv=str(labels_csv),
        transcripts_en_json=str(transcripts_de_json),
        partition="Train",
        modalities=['text_en'],  # folosim text_en ca workaround pentru germană
    )
    print("\n2. Loading Validation dataset...")
    val_dataset_msp = MSP_Podcast_Dataset(
        audio_root=str(data_dir / "Audios"),
        labels_csv=str(labels_csv),
        transcripts_en_json=str(transcripts_de_json),
        partition="Development",
        modalities=['text_en'],  # folosim text_en ca workaround pentru germană
    )
    print(f"\nData loaded successfully!")
    print(train_dataset_msp[0])
    print(f"   Train: {len(train_dataset_msp)} samples")
    print(f"   Val: {len(val_dataset_msp)} samples\n")
    trainer = TextOnlyTrainerDE()
    train_dataset = TextEncoderDataset(train_dataset_msp, trainer.tokenizer)
    val_dataset = TextEncoderDataset(val_dataset_msp, trainer.tokenizer)
    trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        checkpoint_dir=checkpoint_dir,
        num_epochs=5,
        batch_size=256,
        learning_rate=2e-4,
        lora_r=16,
        lora_alpha=32,
        gradient_accumulation_steps=1,
        class_weights=train_dataset_msp.get_class_weights(
           all_train_labels=train_dataset_msp.metadata['label_id'].values,
           device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
    )

    # Rulează automat scriptul pentru franceză după terminarea celui pentru germană
    import subprocess
    import sys
    print("\nRulăm automat train_roberta_text_fr.py...")
    subprocess.run([sys.executable, str(Path(__file__).parent / "train_roberta_text_fr.py")], check=True)

if __name__ == "__main__":
    main()
