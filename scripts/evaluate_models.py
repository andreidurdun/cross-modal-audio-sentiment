"""
Script de evaluare pentru compararea modelului original cu cel fine-tuned.
Versiune simplificată care folosește modulul de încărcare date optimizat.
"""
from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import AutoPeftModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from tqdm import tqdm

# Import optimizat pentru încărcare date
from src.data.dataset import MSP_Podcast_Dataset


class ModelEvaluator:
    """Evaluator pentru compararea modelelor."""

    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.label_names = ["unsatisfied", "neutral", "satisfied"]

    def load_original_model(self, model_name: str):
        """Încarcă modelul original pre-trained."""
        print(f"Loading original model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model = model.to(self.device)
        model.eval()
        return tokenizer, model

    def load_finetuned_model(self, model_path: Path):
        """Încarcă modelul fine-tuned cu LoRA."""
        print(f"Loading fine-tuned model from: {model_path}")
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Fine-tuned model not found at: {model_path}\n"
                f"Please train the model first using train_roberta_text.py"
            )
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoPeftModelForSequenceClassification.from_pretrained(
            model_path,
            device_map=self.device,
        )
        model.eval()
        return tokenizer, model

    @torch.no_grad()
    def predict_batch(self, model, tokenizer, texts: list[str], batch_size: int = 32) -> list[int]:
        """Predicții în batch-uri pentru eficiență."""
        predictions = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            outputs = model(**inputs)
            batch_preds = outputs.logits.argmax(dim=-1).cpu().tolist()
            predictions.extend(batch_preds)
        
        return predictions

    def evaluate_model(
        self,
        model,
        tokenizer,
        texts: list[str],
        labels: list[int],
        model_name: str = "Model"
    ) -> dict:
        """Evaluează un model și returnează metrici."""
        print(f"\nEvaluating {model_name}...")
        
        # Predicții
        predictions = self.predict_batch(model, tokenizer, texts, batch_size=32)
        
        # Metrici
        accuracy = accuracy_score(labels, predictions)
        f1_macro = f1_score(labels, predictions, average='macro')
        f1_weighted = f1_score(labels, predictions, average='weighted')
        f1_per_class = f1_score(labels, predictions, average=None)
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        # Classification report
        report = classification_report(
            labels, 
            predictions, 
            target_names=self.label_names,
            digits=4
        )
        
        results = {
            "model_name": model_name,
            "accuracy": float(accuracy),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
            "f1_per_class": {
                self.label_names[i]: float(f1_per_class[i])
                for i in range(len(self.label_names))
            },
            "confusion_matrix": cm.tolist(),
            "classification_report": report
        }
        
        return results

    def compare_models(
        self,
        original_results: dict,
        finetuned_results: dict
    ):
        """Afișează comparație între modele."""
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)
        
        print(f"\n{'Metric':<20} {'Original':<15} {'Fine-tuned':<15} {'Improvement':<15}")
        print("-" * 70)
        
        # Accuracy
        orig_acc = original_results['accuracy']
        ft_acc = finetuned_results['accuracy']
        improvement = ft_acc - orig_acc
        print(f"{'Accuracy':<20} {orig_acc:<15.4f} {ft_acc:<15.4f} {improvement:+.4f}")
        
        # F1 Macro
        orig_f1 = original_results['f1_macro']
        ft_f1 = finetuned_results['f1_macro']
        improvement = ft_f1 - orig_f1
        print(f"{'F1 Macro':<20} {orig_f1:<15.4f} {ft_f1:<15.4f} {improvement:+.4f}")
        
        # F1 Weighted
        orig_f1w = original_results['f1_weighted']
        ft_f1w = finetuned_results['f1_weighted']
        improvement = ft_f1w - orig_f1w
        print(f"{'F1 Weighted':<20} {orig_f1w:<15.4f} {ft_f1w:<15.4f} {improvement:+.4f}")
        
        print("\nPer-Class F1 Scores:")
        print("-" * 70)
        for label in self.label_names:
            orig_f1 = original_results['f1_per_class'][label]
            ft_f1 = finetuned_results['f1_per_class'][label]
            improvement = ft_f1 - orig_f1
            print(f"  {label:<18} {orig_f1:<15.4f} {ft_f1:<15.4f} {improvement:+.4f}")
        
        print("\n" + "="*80)


def main():
    """Main evaluation function."""
    
    # Paths
    data_dir = Path("MSP_Podcast")
    labels_csv = data_dir / "Labels" / "labels_consensus.csv"
    transcripts_en_json = data_dir / "Transcription_en.json"
    finetuned_model_path = Path("checkpoints/roberta_text_en/best_model")
    
    # Model names
    original_model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    
    # Verificare fișiere
    if not labels_csv.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_csv}")
    if not transcripts_en_json.exists():
        raise FileNotFoundError(f"Transcripts JSON not found: {transcripts_en_json}")
    
    print("="*80)
    print("MODEL EVALUATION AND COMPARISON")
    print("="*80)
    
    # Load test data
    print("\nLoading test data...")
    test_dataset = MSP_Podcast_Dataset(
        audio_root=str(data_dir / "Audios"),
        labels_csv=str(labels_csv),
        transcripts_en_json=str(transcripts_en_json),
        partition="Test1",
        modalities=['text_en'],
        use_cache=True,
        max_workers=8,
    )
    test_texts = [test_dataset[idx]['text_en'] for idx in range(len(test_dataset))]
    test_labels = [test_dataset[idx]['label'] for idx in range(len(test_dataset))]
    
    print(f"\n✅ Test data loaded: {len(test_texts)} samples\n")
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Evaluate original model
    print("\n" + "="*80)
    print("1. ORIGINAL MODEL")
    print("="*80)
    tokenizer_orig, model_orig = evaluator.load_original_model(original_model_name)
    original_results = evaluator.evaluate_model(
        model_orig, 
        tokenizer_orig, 
        test_texts, 
        test_labels,
        model_name="Original RoBERTa"
    )
    
    print("\nOriginal Model Results:")
    print(original_results['classification_report'])
    
    # Evaluate fine-tuned model
    print("\n" + "="*80)
    print("2. FINE-TUNED MODEL")
    print("="*80)
    tokenizer_ft, model_ft = evaluator.load_finetuned_model(finetuned_model_path)
    finetuned_results = evaluator.evaluate_model(
        model_ft,
        tokenizer_ft,
        test_texts,
        test_labels,
        model_name="Fine-tuned RoBERTa + LoRA"
    )
    
    print("\nFine-tuned Model Results:")
    print(finetuned_results['classification_report'])
    
    # Compare models
    evaluator.compare_models(original_results, finetuned_results)
    
    # Save results
    results_file = Path("evaluation_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            "original": original_results,
            "finetuned": finetuned_results,
            "test_samples": len(test_texts)
        }, f, indent=2)
    
    print(f"\n✅ Results saved to: {results_file}")


if __name__ == "__main__":
    main()
