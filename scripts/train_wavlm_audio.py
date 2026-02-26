"""
WavLM Fine-tuning cu Standard LoRA (FP16/BF16) pentru clasificare audio.
Script optimizat pentru acuratețe maximă și prevenirea OOM pe RTX 4060.
Model: microsoft/wavlm-base-plus
"""
from pathlib import Path
from typing import Optional
from contextlib import nullcontext
import os

import torch
import sys
import numpy as np
from sklearn.metrics import f1_score
import json
import warnings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")

from torch.utils.data import DataLoader, Dataset
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
from tqdm import tqdm

from src.data.dataset import MSP_Podcast_Dataset

class AudioWaveLMDataset(Dataset):
    def __init__(self, msp_dataset: MSP_Podcast_Dataset, feature_extractor):
        self.msp_dataset = msp_dataset
        self.feature_extractor = feature_extractor
        self.sample_rate = 16000

    def __len__(self):
        return len(self.msp_dataset)

    def __getitem__(self, idx):
        sample = self.msp_dataset[idx]
        audio = sample['audio'].numpy() if isinstance(sample['audio'], torch.Tensor) else sample['audio']
        
        # Verifică și curăță audio-ul de valori invalide
        if np.isnan(audio).any() or np.isinf(audio).any():
            audio = np.zeros_like(audio)

        # Trunchiere la 6 secunde
        MAX_SECONDS = 6 
        max_samples = self.sample_rate * MAX_SECONDS
        
        if audio.shape[-1] > max_samples:
            audio = audio[..., :max_samples]
        
        inputs = self.feature_extractor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        )
        
        return {
            'input_values': inputs['input_values'].squeeze(0),
            'labels': torch.tensor(sample['label'], dtype=torch.long),
        }

class AudioCollator:
    def __call__(self, batch):
        input_features = [sample['input_values'] for sample in batch]
        labels = torch.stack([sample['labels'] for sample in batch])
        
        batch_size = len(input_features)
        max_length = max(feature.shape[-1] for feature in input_features)
        
        # Generăm tensori pentru input și mască de atenție (padding dinamic pe batch)
        padded_inputs = torch.zeros((batch_size, max_length), dtype=torch.float32)
        attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
        
        for i, feature in enumerate(input_features):
            seq_len = feature.shape[-1]
            padded_inputs[i, :seq_len] = feature
            attention_mask[i, :seq_len] = 1 # 1 pentru audio real, 0 pentru padding
            
        return {
            'input_values': padded_inputs,
            'attention_mask': attention_mask,
            'labels': labels
        }

class AudioTrainerWavLM:
    def __init__(self, model_name: str = "microsoft/wavlm-base-plus", num_labels: int = 3, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.num_labels = num_labels
        self.id2label = {0: "unsatisfied", 1: "neutral", 2: "satisfied"}
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

    def setup_model_with_lora(self, lora_r: int = 16, lora_alpha: int = 32):
        print("Loading model in FP32 for numerical stability...")
        model = AutoModelForAudioClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
            torch_dtype=torch.float32,
            ignore_mismatched_sizes=True,
        )

        model.config.use_cache = False

        print("Configuring LoRA...")
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.1,
            bias="none",
            # CORECRURA 1: "all-linear" atacă toate straturile dense, maximizând capacitatea de adaptare
            target_modules="all-linear", 
            modules_to_save=["classifier", "projector"], 
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        return model.to(self.device)

    def train_epoch(self, model, train_loader, optimizer, scheduler, scaler, use_amp, gradient_accumulation_steps=1) -> float:
        model.train()
        total_loss = 0.0
        valid_steps = 0
        consecutive_nans = 0
        MAX_CONSECUTIVE_NANS = 100 

        # Ponderile pentru clase - vitale pentru MSP-Podcast
        class_weights = torch.tensor([1.02, 1.00, 1.60], dtype=torch.float32).to(self.device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

        progress_bar = tqdm(train_loader, desc="Training", position=0)
        optimizer.zero_grad(set_to_none=True)
        
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        for step, batch in enumerate(progress_bar):
            input_values = batch['input_values'].to(self.device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)
            
            if torch.isnan(input_values).any() or torch.isinf(input_values).any():
                consecutive_nans += 1
                if consecutive_nans > MAX_CONSECUTIVE_NANS:
                    print(f"\n❌ ABORT: Too many consecutive NaN batches ({consecutive_nans}).")
                    return float('nan')
                continue
            else:
                consecutive_nans = 0 
            
            autocast_ctx = torch.amp.autocast(dtype=amp_dtype, device_type='cuda') if use_amp else nullcontext()

            with autocast_ctx:
                # 1. Rulăm modelul FĂRĂ labels pentru a extrage logiții puri
                outputs = model(
                    input_values=input_values, 
                    attention_mask=attention_mask
                )
                
                # 2. Asigurăm forma corectă [batch_size, num_classes] și cast la FP32 pentru stabilitate
                logits = outputs.logits.float().view(-1, self.num_labels)
                labels_flat = labels.view(-1)
                
                # 3. Calculăm loss-ul ponderat manual
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
    def evaluate(self, model, val_loader, use_amp: bool) -> tuple:
        model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []

        # Păstrăm ponderile și la validare pentru a avea un Loss comparabil
        class_weights = torch.tensor([1.02, 1.00, 1.60], dtype=torch.float32).to(self.device)
        val_loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

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
                
                # Același procedeu de siguranță pentru forma tensorilor
                logits = outputs.logits.float().view(-1, self.num_labels)
                labels_flat = labels.view(-1)
                
                loss = val_loss_fn(logits, labels_flat)
                
            total_loss += loss.item()

            predictions = logits.argmax(dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels_flat.cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
        f1_macro = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
        return avg_loss, accuracy, f1_macro
    
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
    ):
        print("="*80)
        print("Audio Training with AMP - WavLM")
        print("="*80)

        model = self.setup_model_with_lora(lora_r=lora_r, lora_alpha=lora_alpha)

        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        else:
            use_amp = False

        collator = AudioCollator()
        
        # Num_workers a fost lăsat la 0 pentru a preveni crash-uri pe Windows la procesare audio masivă
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=0,
            pin_memory=True if self.device == "cuda" else False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            collate_fn=collator,
            num_workers=0,
            pin_memory=True if self.device == "cuda" else False,
        )
        
        if test_dataset is not None:
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size * 2,
                shuffle=False,
                collate_fn=collator,
                num_workers=0,
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
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

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
            )
            print(f"Train Loss: {train_loss:.4f}")

            val_loss, val_acc, val_f1 = self.evaluate(model, val_loader, use_amp)
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1 Macro: {val_f1:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_path = checkpoint_dir / "best_model"
                print(f"✅ New best model! F1 Macro: {val_f1:.4f} - Saving...")
                
                model.save_pretrained(best_model_path)
                self.feature_extractor.save_pretrained(best_model_path)

        if test_loader is not None:
            print("\nEvaluating on test set...")
            test_loss, test_acc, test_f1 = self.evaluate(model, test_loader, use_amp)
            print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test F1 Macro: {test_f1:.4f}")
            return {'best_val_f1': best_val_f1, 'test_loss': test_loss, 'test_acc': test_acc, 'test_f1': test_f1}
        else:
            print("\nNo test dataset provided. Skipping test evaluation.")
            return {'best_val_f1': best_val_f1}

def main():
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
        use_cache=True,
    )

    print("\n2. Loading Validation dataset...")
    val_dataset_msp = MSP_Podcast_Dataset(
        audio_root=str(data_dir / "Audios"),
        labels_csv=str(labels_csv),
        partition="Development",
        modalities=['audio'],
        use_cache=True,
    )

    print(f"\nData loaded successfully!")
    print(f"   Train: {len(train_dataset_msp)} samples")
    print(f"   Val: {len(val_dataset_msp)} samples")

    feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
    
    train_dataset = AudioWaveLMDataset(train_dataset_msp, feature_extractor)
    val_dataset = AudioWaveLMDataset(val_dataset_msp, feature_extractor)
    
    trainer = AudioTrainerWavLM()
    
    # CORECRURA 2: LR 3e-4 si Batch Size sigur pentru VRAM
    results = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=None,
        checkpoint_dir=checkpoint_dir,
        num_epochs=5,
        batch_size=16,                  # Mai mic, previne aglomerarea RAM-ului
        gradient_accumulation_steps=4, # 8x8 = 64 Effective Batch Size
        learning_rate=3e-4,            # Optim pentru LoRA
        lora_r=16,
        lora_alpha=32,
        use_amp=True,
    )

    results_path = checkpoint_dir / "best_model" / "training_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}")
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)

if __name__ == "__main__":
    main()