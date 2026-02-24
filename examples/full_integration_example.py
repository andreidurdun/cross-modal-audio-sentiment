"""
Exemplu complet de integrare: Încărcare backbones + CCMT + Predicție
Demonstrează cum să folosești întreaga arhitectură multimodală.
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.models import (
    load_all_backbones,
    load_full_multimodal_model,
    MultimodalEmotionModel,
)


def example_1_load_individual_backbones():
    """Exemplu 1: Încarcă backbones individuali și folosește-i separat."""
    print("\n" + "="*80)
    print("EXEMPLU 1: Încărcare Backbones Individuali")
    print("="*80)
    
    # Încarcă toate backbones
    backbones = load_all_backbones(
        text_en_checkpoint="checkpoints/roberta_text_en",
        text_es_checkpoint="checkpoints/roberta_text_es",
        audio_checkpoint="checkpoints/wavlm_audio",
        freeze=True,
        projection_dim=256,  # Dimensiune comună pentru toate
    )
    
    # Test text EN
    texts_en = ["This is a happy sentence", "I am very sad today"]
    with torch.no_grad():
        embeddings_en = backbones['text_en'](texts_en)
    print(f"\n✓ Text EN embeddings: {embeddings_en.shape}")
    
    # Test text ES  
    texts_es = ["Esto es una frase feliz", "Estoy muy triste hoy"]
    with torch.no_grad():
        embeddings_es = backbones['text_es'](texts_es)
    print(f"✓ Text ES embeddings: {embeddings_es.shape}")
    
    # Test audio (necesită waveforms)
    # Se presupune că ai audio procesate cu feature_extractor
    dummy_audio = torch.randn(2, 16000 * 3)  # 2 samples, 3 seconds
    with torch.no_grad():
        embeddings_audio = backbones['audio'](dummy_audio)
    print(f"✓ Audio embeddings: {embeddings_audio.shape}")
    
    print(f"\n✓ Toate backbones funcționează corect!")


def example_2_load_full_model():
    """Exemplu 2: Încarcă modelul complet end-to-end."""
    print("\n" + "="*80)
    print("EXEMPLU 2: Încărcare Model Complet")
    print("="*80)
    
    # Încarcă modelul complet cu toate backbones integrate
    model = load_full_multimodal_model(
        text_en_checkpoint="checkpoints/roberta_text_en",
        text_es_checkpoint="checkpoints/roberta_text_es",
        audio_checkpoint="checkpoints/wavlm_audio",
        num_classes=3,
        ccmt_dim=1024,
        num_patches_per_modality=100,
        ccmt_depth=6,
        ccmt_heads=8,
        freeze_backbones=True,
        projection_dim=256,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    return model


def example_3_inference_with_embeddings(model):
    """Exemplu 3: Predicție folosind embeddings pre-calculate."""
    print("\n" + "="*80)
    print("EXEMPLU 3: Predicție cu Embeddings Pre-calculate")
    print("="*80)
    
    batch_size = 4
    
    # Simulare embeddings pre-calculate (256 dim - de la backbones cu projection)
    text_en_emb = torch.randn(batch_size, 256, device=model.ccmt.mlp_head[0].weight.device)
    text_es_emb = torch.randn(batch_size, 256, device=model.ccmt.mlp_head[0].weight.device)
    audio_emb = torch.randn(batch_size, 256, device=model.ccmt.mlp_head[0].weight.device)
    
    # Predicție
    model.eval()
    with torch.no_grad():
        predictions, classes = model.predict(
            text_en_emb=text_en_emb,
            text_es_emb=text_es_emb,
            audio_emb=audio_emb,
        )
    
    print(f"\n✓ Predictions shape: {predictions.shape}")
    print(f"✓ Predicted classes: {classes}")
    print(f"✓ Sample probabilities:\n{predictions[:2]}")
    
    # Interpretare clase
    class_names = ["unsatisfied", "neutral", "satisfied"]
    for i, cls in enumerate(classes[:3]):
        print(f"\n  Sample {i}: {class_names[cls.item()]} (confidence: {predictions[i, cls].item():.3f})")


def example_4_end_to_end_inference(model):
    """Exemplu 4: Predicție end-to-end cu input text și audio raw."""
    print("\n" + "="*80)
    print("EXEMPLU 4: Predicție End-to-End")
    print("="*80)
    
    # Input text
    texts_en = [
        "I am extremely happy today!",
        "This is terrible, I feel awful.",
    ]
    texts_es = [
        "¡Estoy extremadamente feliz hoy!",
        "Esto es terrible, me siento fatal.",
    ]
    
    # Audio dummy (în practică ar fi waveforms reale)
    audio = torch.randn(2, 16000 * 3).to(model.ccmt.mlp_head[0].weight.device)
    
    # Predicție end-to-end
    model.eval()
    with torch.no_grad():
        predictions, classes = model.predict(
            text_en=texts_en,
            text_es=texts_es,
            audio=audio,
        )
    
    print(f"\n✓ Predictions shape: {predictions.shape}")
    print(f"✓ Predicted classes: {classes}")
    
    class_names = ["unsatisfied", "neutral", "satisfied"]
    for i in range(len(texts_en)):
        cls = classes[i].item()
        prob = predictions[i, cls].item()
        print(f"\n  Sample {i}:")
        print(f"    EN: {texts_en[i]}")
        print(f"    ES: {texts_es[i]}")
        print(f"    Prediction: {class_names[cls]} (confidence: {prob:.3f})")


def example_5_training_loop():
    """Exemplu 5: Loop de training simplificat."""
    print("\n" + "="*80)
    print("EXEMPLU 5: Training Loop (Simplificat)")
    print("="*80)
    
    # Încarcă model
    model = load_full_multimodal_model(
        freeze_backbones=True,  # Backbones frozen, trainăm doar fusion + CCMT
        projection_dim=256,
    )
    
    device = next(model.parameters()).device
    
    # Optimizer doar pentru parametrii trainable
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=0.01,
    )
    
    # Loss function
    criterion = torch.nn.CrossEntropyLoss()
    
    # Dummy data
    batch_size = 8
    text_en_emb = torch.randn(batch_size, 256).to(device)
    text_es_emb = torch.randn(batch_size, 256).to(device)
    audio_emb = torch.randn(batch_size, 256).to(device)
    labels = torch.randint(0, 3, (batch_size,)).to(device)
    
    # Training step
    model.train()
    
    # Forward
    predictions = model(
        text_en_emb=text_en_emb,
        text_es_emb=text_es_emb,
        audio_emb=audio_emb,
    )
    
    # Loss
    loss = criterion(predictions, labels)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"\n✓ Training step completed")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Trainable parameters: {model.get_trainable_parameters():,}")


def main():
    """Run toate exemplele."""
    print("\n" + "="*80)
    print("ARHITECTURĂ MULTIMODALĂ COMPLETĂ - EXEMPLE DE INTEGRARE")
    print("="*80)
    
    # Exemplu 1: Backbones individuali
    example_1_load_individual_backbones()
    
    # Exemplu 2: Model complet
    model = example_2_load_full_model()
    
    # Exemplu 3: Predicție cu embeddings
    example_3_inference_with_embeddings(model)
    
    # Exemplu 4: Predicție end-to-end
    example_4_end_to_end_inference(model)
    
    # Exemplu 5: Training loop
    example_5_training_loop()
    
    print("\n" + "="*80)
    print("✅ TOATE EXEMPLELE AU FOST RULATE CU SUCCES!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
