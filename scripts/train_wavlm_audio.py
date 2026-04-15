from pathlib import Path
from typing import Optional
from contextlib import nullcontext
import os

import torch
import sys
import numpy as np
from sklearn.metrics import f1_score
import json
import time
import matplotlib.pyplot as plt
import warnings

try:
    from scripts._bootstrap import project_root
except ModuleNotFoundError:
    from _bootstrap import project_root

PROJECT_ROOT = project_root()

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*use_reentrant parameter should be passed explicitly.*")

from torch.utils.data import DataLoader
from transformers import (
    AutoModelForAudioClassification, 
    AutoFeatureExtractor,
    get_linear_schedule_with_warmup,
)
from peft import (
    LoraConfig, 
    TaskType, 
    get_peft_model, 
)
from tqdm.auto import tqdm

from src.data.dataset import MSP_Podcast_Dataset
from src.data.audio_datasets import AudioWaveLMDataset, AudioCollator
from src.utils.config import get_training_config

class AudioTrainerWavLM:
    def __init__(self, model_name: str = "microsoft/wavlm-base-plus", num_labels: int = 3, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.num_labels = num_labels
        self.id2label = {0: "unsatisfied", 1: "neutral", 2: "satisfied"}
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

    def setup_model_with_lora(self, lora_r: int = 16, lora_alpha: int = 32):
        model_dtype = torch.float32

        print("Loading base model in native precision...")
        model = AutoModelForAudioClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True,
            torch_dtype=model_dtype,
        )

        model.config.use_cache = False
        # Gradient checkpointing disabled pentru viteza (trade-off: mai mult VRAM)
        # model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        print("Configuring LoRA...")
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.1,
            bias="none",
            target_modules="all-linear", 
            modules_to_save=["classifier", "projector"], 
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        return model.to(self.device)

    def train_epoch(
        self,
        model,
        train_loader,
        optimizer,
        scheduler,
        scaler,
        use_amp,
        gradient_accumulation_steps=1,
        class_weights: Optional[torch.Tensor] = None,
    ) -> float:
        model.train()
        total_loss = 0.0
        valid_steps = 0
        consecutive_nans = 0
        MAX_CONSECUTIVE_NANS = 100

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
            mininterval=0.5,   # update max de 2 ori/s
        )

        optimizer.zero_grad(set_to_none=True)
        
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        for step, batch in enumerate(progress_bar):
            input_values = batch['input_values'].to(self.device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)
            
            if torch.isnan(input_values).any() or torch.isinf(input_values).any():
                consecutive_nans += 1
                if consecutive_nans > MAX_CONSECUTIVE_NANS:
                    print(f"\nABORT: Too many consecutive NaN batches ({consecutive_nans}).")
                    return float('nan')
                continue
            else:
                consecutive_nans = 0 
            
            autocast_ctx = torch.amp.autocast(dtype=amp_dtype, device_type='cuda') if use_amp else nullcontext()

            with autocast_ctx:
                
                outputs = model(
                    input_values=input_values, 
                    attention_mask=attention_mask
                )
                
                #asiguram forma corecta a tensorilor pentru calculul loss-ului, indiferent de batch size sau lungimea secventei
                logits = outputs.logits.float().view(-1, self.num_labels)
                labels_flat = labels.view(-1)
                
             
                loss = loss_fn(logits, labels_flat)
            
            if torch.isnan(loss) or torch.isinf(loss):
                consecutive_nans += 1
                if consecutive_nans > MAX_CONSECUTIVE_NANS:
                    return float('nan')
                continue
            else:
                consecutive_nans = 0
            
            loss = loss / gradient_accumulation_steps
            
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                has_nan_grad = False
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            has_nan_grad = True
                            break
                
                if has_nan_grad:
                    optimizer.zero_grad(set_to_none=True)
                    consecutive_nans += 1
                    continue
                
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            actual_loss = loss.item() * gradient_accumulation_steps
            total_loss += actual_loss
            valid_steps += 1
            progress_bar.set_postfix({"loss": f"{actual_loss:.4f}"})

        avg_loss = total_loss / valid_steps if valid_steps > 0 else float('nan')
        return avg_loss

    @torch.no_grad()
    def evaluate(
        self,
        model,
        val_loader,
        use_amp: bool,
        class_weights: Optional[torch.Tensor] = None,
    ) -> tuple:
        model.eval()
        total_loss = 0.0
        total_samples = 0
        all_predictions = []
        all_labels = []

        if class_weights is not None:
            val_loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
        else:
            val_loss_fn = torch.nn.CrossEntropyLoss()

        progress_bar = tqdm(val_loader, desc="Evaluating", position=0, leave=False)
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        for batch in progress_bar:
            input_values = batch['input_values'].to(self.device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)

            autocast_ctx = torch.amp.autocast(dtype=amp_dtype, device_type='cuda') if use_amp else nullcontext()
            with autocast_ctx:
                outputs = model(
                    input_values=input_values, 
                    attention_mask=attention_mask
                )
                
    
                logits = outputs.logits.float().view(-1, self.num_labels)
                labels_flat = labels.view(-1)
                
                loss = val_loss_fn(logits, labels_flat)

            batch_size = labels_flat.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            predictions = logits.argmax(dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels_flat.cpu().numpy())

        avg_loss = total_loss / total_samples
        accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
        f1_macro = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
        return avg_loss, accuracy, f1_macro

    def _plot_training_curves(self, train_losses, val_losses, checkpoint_dir):
        """Genereaza si salveaza grafic cu loss-urile de antrenare si validare"""
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
        test_dataset,
        checkpoint_dir,
        num_epochs=5,
        batch_size=8,
        learning_rate=3e-4,
        lora_r=16,
        lora_alpha=32,
        gradient_accumulation_steps=8,
        use_amp=True,
        num_workers=4,
        class_weights: Optional[torch.Tensor] = None,
    ):
        print("="*80)
        print("Audio Training with AMP - WavLM")
        print("="*80)

        #incepe tracking timpului
        start_time = time.time()

        model = self.setup_model_with_lora(lora_r=lora_r, lora_alpha=lora_alpha)

        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        else:
            use_amp = False

        collator = AudioCollator()

        #pt widows pastram 0 worker pentru a evita problemele de multiprocessing
        effective_num_workers = 0 if os.name == "nt" else max(0, num_workers)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=effective_num_workers,
            pin_memory=True if self.device == "cuda" else False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            collate_fn=collator,
            num_workers=effective_num_workers,
            pin_memory=True if self.device == "cuda" else False,
        )
        
        if test_dataset is not None:
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size * 2,
                shuffle=False,
                collate_fn=collator,
                num_workers=effective_num_workers,
                pin_memory=True if self.device == "cuda" else False,
            )
        else:
            test_loader = None

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        
        total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

        best_val_f1 = 0.0
        best_val_loss = float('inf')
        best_val_acc = 0.0
        last_train_loss = 0.0
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        #liste pentru tracking loss-urile
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

            train_loss = self.train_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                scaler,
                use_amp,
                gradient_accumulation_steps,
                class_weights,
            )
            train_losses.append(train_loss)
            last_train_loss = train_loss
            print(f"Train Loss: {train_loss:.4f}")

            val_loss, val_acc, val_f1 = self.evaluate(model, val_loader, use_amp, class_weights)
            val_losses.append(val_loss)
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1 Macro: {val_f1:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_model_path = checkpoint_dir / "best_model"
                print(f"New best model! F1 Macro: {val_f1:.4f} - Saving...")

                # Salvam modelul complet cand e posibil (compatibil cu load_audio_backbone)
                try:
                    merged_model = model.merge_and_unload() if hasattr(model, "merge_and_unload") else model
                    merged_model.save_pretrained(best_model_path)
                    print("Saved merged full model for downstream backbone loading.")
                except Exception as exc:
                    print(f"[WARN] Could not merge LoRA adapters ({exc}). Saving adapter-only checkpoint.")
                    model.save_pretrained(best_model_path)

                self.feature_extractor.save_pretrained(best_model_path)

        #calculeaza timp total
        end_time = time.time()
        total_time_seconds = end_time - start_time
        total_time_minutes = total_time_seconds / 60
        total_time_hours = total_time_minutes / 60

        print("\n" + "="*80)
        print(f"Training Complete!")
        print(f"Best Validation F1 Macro: {best_val_f1:.4f}")
        print(f"Model saved to: {checkpoint_dir / 'best_model'}")
        print("="*80)

        #salveaza metrici antrenarii in JSON
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

        #genereaza si salveaza grafic cu loss-urile
        self._plot_training_curves(train_losses, val_losses, checkpoint_dir)

        if test_loader is not None:
            print("\nEvaluating on test set...")
            test_loss, test_acc, test_f1 = self.evaluate(model, test_loader, use_amp, class_weights)
            print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test F1 Macro: {test_f1:.4f}")

def main():
    train_config = get_training_config(
        "wavlm_audio",
        PROJECT_ROOT / "configs" / "training_config.json",
    )

    data_dir = Path("MSP_Podcast")
    labels_csv = data_dir / "Labels" / "labels_consensus.csv"
    checkpoint_dir = Path("checkpoints/wavlm_audio")
    
    if not labels_csv.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_csv}")
    
    print("="*80)
    print("Loading MSP-Podcast Audio Data with WavLM")
    print("="*80)

    print("\n1. Loading Train dataset...")
    train_dataset_msp = MSP_Podcast_Dataset(
        audio_root=str(data_dir / "Audios"),
        labels_csv=str(labels_csv),
        partition="Train",
        modalities=['audio'],
    )

    print("\n2. Loading Validation dataset...")
    val_dataset_msp = MSP_Podcast_Dataset(
        audio_root=str(data_dir / "Audios"),
        labels_csv=str(labels_csv),
        partition="Development",
        modalities=['audio'],
    )

    print(f"\nData loaded successfully!")
    print(f"   Train: {len(train_dataset_msp)} samples")
    print(f"   Val: {len(val_dataset_msp)} samples")

    feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
    
    train_dataset = AudioWaveLMDataset(
        train_dataset_msp,
        feature_extractor,
        max_seconds=5,
        do_resample=False,
        label_key="label",
        include_attention_mask=False,
    )
    val_dataset = AudioWaveLMDataset(
        val_dataset_msp,
        feature_extractor,
        max_seconds=5,
        do_resample=False,
        label_key="label",
        include_attention_mask=False,
    )
    
    trainer = AudioTrainerWavLM()
    trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=None,
        checkpoint_dir=checkpoint_dir,
        num_epochs=int(train_config["num_epochs"]),
        batch_size=int(train_config["batch_size"]),
        gradient_accumulation_steps=int(train_config["gradient_accumulation_steps"]),
        learning_rate=float(train_config["learning_rate"]),
        lora_r=int(train_config["lora_r"]),
        lora_alpha=int(train_config["lora_alpha"]),
        use_amp=bool(train_config["use_amp"]),
        class_weights=train_dataset_msp.get_class_weights(
            all_train_labels=train_dataset_msp.metadata['label_id'].values,
            device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ),
    )

if __name__ == "__main__":
    main()