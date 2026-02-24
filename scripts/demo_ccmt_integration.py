"""
Script exemplu: Cum să folosești modelul multimodal CCMT complet.
Demonstrează încărcarea, predicția și training.
"""
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader

# Import modelul complet
from src.models import load_full_multimodal_model, MultimodalEmotionModel


def example_1_load_and_predict():
    """Exemplu 1: Încarcă modelul și fă predicții simple."""
    print("\n" + "="*70)
    print("EXEMPLU 1: Încărcare model și predicție")
    print("="*70)
    
    # Încarcă modelul cu backbones pretrenate
    model = load_full_multimodal_model(
        text_en_checkpoint="checkpoints/roberta_text_en",
        text_es_checkpoint="checkpoints/roberta_text_es",
        num_classes=3,
        ccmt_dim=1024,
        num_patches_per_modality=100,
        ccmt_depth=6,
        ccmt_heads=8,
        freeze_backbones=True,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Pregătește date demo
    texts_en = ["I feel great today!", "This is terrible"]
    texts_es = ["Me siento genial hoy!", "Esto es terrible"]
    
    # Audio dummy (în practică ar veni de la dataset)
    batch_size = len(texts_en)
    audio_emb = torch.randn(batch_size, 768).to(model.text_en_backbone.model.device)
    
    # Predicție
    predictions, predicted_classes = model.predict(
        text_en=texts_en,
        text_es=texts_es,
        audio_emb=audio_emb,
    )
    
    print(f"\nPredicții (probabilități):")
    for i, (pred, cls) in enumerate(zip(predictions, predicted_classes)):
        print(f"  Sample {i}: {pred.cpu().numpy()} → Class {cls.item()}")
    
    print("\n✅ Predicție completă!")


def example_2_precompute_embeddings():
    """Exemplu 2: Pre-calculate embeddings pentru eficiență."""
    print("\n" + "="*70)
    print("EXEMPLU 2: Pre-calculare embeddings pentru inferență rapidă")
    print("="*70)
    
    model = load_full_multimodal_model(
        text_en_checkpoint="checkpoints/roberta_text_en",
        text_es_checkpoint="checkpoints/roberta_text_es",
        freeze_backbones=True,
    )
    
    device = next(model.parameters()).device
    
    # Simulăm un batch de embeddings pre-calculate
    batch_size = 16
    text_en_emb = torch.randn(batch_size, 768).to(device)
    text_es_emb = torch.randn(batch_size, 768).to(device)
    audio_emb = torch.randn(batch_size, 768).to(device)
    
    # Inferență DOAR prin fusion + CCMT (mult mai rapid!)
    predictions = model(
        text_en_emb=text_en_emb,
        text_es_emb=text_es_emb,
        audio_emb=audio_emb,
    )
    
    print(f"\nPredicții shape: {predictions.shape}")
    print(f"Media predicții per clasă: {predictions.mean(dim=0)}")
    print("\n✅ Inferență cu embeddings pre-calculate completă!")


def example_3_training_setup():
    """Exemplu 3: Setup pentru training."""
    print("\n" + "="*70)
    print("EXEMPLU 3: Setup pentru training CCMT")
    print("="*70)
    
    # Încarcă modelul cu backbones frozen
    model = load_full_multimodal_model(
        text_en_checkpoint="checkpoints/roberta_text_en",
        text_es_checkpoint="checkpoints/roberta_text_es",
        freeze_backbones=True,  # Antrenăm doar fusion + CCMT
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Configurare optimizer - doar parametrii trainable
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=0.01)
    
    # Loss function
    criterion = nn.BCELoss()  # CCMT folosește Sigmoid în output
    
    print(f"\nParametri trainable: {model.get_trainable_parameters():,}")
    print(f"Optimizer are {len(trainable_params)} grupuri de parametri")
    
    # Simulare training step
    model.train()
    
    batch_size = 8
    device = next(model.parameters()).device
    
    # Batch simulat de embeddings
    text_en_emb = torch.randn(batch_size, 768).to(device)
    text_es_emb = torch.randn(batch_size, 768).to(device)
    audio_emb = torch.randn(batch_size, 768).to(device)
    labels = torch.randint(0, 3, (batch_size,)).to(device)
    
    # One-hot encode pentru BCE
    labels_onehot = torch.zeros(batch_size, 3).to(device)
    labels_onehot.scatter_(1, labels.unsqueeze(1), 1.0)
    
    # Forward pass
    predictions = model(
        text_en_emb=text_en_emb,
        text_es_emb=text_es_emb,
        audio_emb=audio_emb,
    )
    
    # Calculate loss
    loss = criterion(predictions, labels_onehot)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"\nTraining step completat:")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Predictions shape: {predictions.shape}")
    
    print("\n✅ Training setup complet!")


def example_4_custom_configuration():
    """Exemplu 4: Configurare personalizată CCMT."""
    print("\n" + "="*70)
    print("EXEMPLU 4: Configurare CCMT personalizată")
    print("="*70)
    
    from src.models import (
        TextENBackbone, 
        TextESBackbone, 
        MultimodalFusionAdapter,
        CascadedCrossModalTransformer
    )
    
    # Construiește modelul component by component
    print("\n1. Creăm backbones personalizate...")
    text_en = TextENBackbone(
        model_name="cardiffnlp/twitter-roberta-base-sentiment-latest",
        freeze=True,
        projection_dim=None,
    )
    
    text_es = TextESBackbone(
        model_name="pysentimiento/robertuito-sentiment-analysis",
        freeze=True,
        projection_dim=None,
    )
    
    print("\n2. Creăm fusion adapter...")
    fusion = MultimodalFusionAdapter(
        text_en_dim=768,
        text_es_dim=768,
        audio_dim=768,
        ccmt_dim=512,  # CCMT mai mic
        num_patches_per_modality=50,  # Mai puține patches
        dropout=0.15,
    )
    
    print("\n3. Creăm CCMT cu configurare custom...")
    ccmt = CascadedCrossModalTransformer(
        num_classes=3,
        num_patches=50 * 3,  # 3 modalități × 50 patches
        dim=512,
        depth=4,  # Mai puține layere
        heads=4,  # Mai puține heads
        mlp_dim=1024,
        dropout=0.15,
    )
    
    print("\n4. Construim modelul complet manual...")
    model = MultimodalEmotionModel(
        text_en_backbone=text_en,
        text_es_backbone=text_es,
        audio_backbone=None,
        text_en_dim=768,
        text_es_dim=768,
        audio_dim=768,
        num_classes=3,
        ccmt_dim=512,
        num_patches_per_modality=50,
        ccmt_depth=4,
        ccmt_heads=4,
        ccmt_mlp_dim=1024,
        freeze_backbones=True,
    )
    
    model.print_parameter_summary()
    
    print("\n✅ Model personalizat creat cu succes!")


def example_5_inference_pipeline():
    """Exemplu 5: Pipeline complet de inferență."""
    print("\n" + "="*70)
    print("EXEMPLU 5: Pipeline COMPLET de inferență")
    print("="*70)
    
    # Încarcă modelul
    model = load_full_multimodal_model(
        text_en_checkpoint="checkpoints/roberta_text_en",
        text_es_checkpoint="checkpoints/roberta_text_es",
        freeze_backbones=True,
    )
    
    model.eval()
    device = next(model.parameters()).device
    
    # Date de test
    test_data = [
        {
            "text_en": "I am very happy with this!",
            "text_es": "Estoy muy feliz con esto!",
            "expected": "satisfied"
        },
        {
            "text_en": "This is terrible, I hate it",
            "text_es": "Esto es terrible, lo odio",
            "expected": "unsatisfied"
        },
        {
            "text_en": "It's okay, nothing special",
            "text_es": "Está bien, nada especial", 
            "expected": "neutral"
        }
    ]
    
    id2label = {0: "unsatisfied", 1: "neutral", 2: "satisfied"}
    
    print("\nPredicții pe date de test:")
    print("-" * 70)
    
    with torch.no_grad():
        for i, sample in enumerate(test_data):
            # Audio simulat (în practică vine din dataset)
            audio_emb = torch.randn(1, 768).to(device)
            
            predictions, predicted_class = model.predict(
                text_en=[sample["text_en"]],
                text_es=[sample["text_es"]],
                audio_emb=audio_emb,
            )
            
            pred_label = id2label[predicted_class.item()]
            probs = predictions[0].cpu().numpy()
            
            print(f"\nSample {i+1}:")
            print(f"  EN: {sample['text_en']}")
            print(f"  ES: {sample['text_es']}")
            print(f"  Predicție: {pred_label}")
            print(f"  Probabilități: {probs}")
            print(f"  Așteptat: {sample['expected']}")
            print(f"  ✓ Corect" if pred_label == sample['expected'] else "  ✗ Greșit")
    
    print("\n" + "-" * 70)
    print("✅ Pipeline de inferență complet!")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("DEMONSTRAȚIE INTEGRARE CCMT ÎN ARHITECTURA MULTIMODALĂ")
    print("="*70)
    
    # Rulează exemplele
    try:
        example_1_load_and_predict()
    except Exception as e:
        print(f"⚠ Exemplul 1 a eșuat (normal dacă nu ai checkpoints): {e}")
    
    try:
        example_2_precompute_embeddings()
    except Exception as e:
        print(f"⚠ Exemplul 2 a eșuat: {e}")
    
    try:
        example_3_training_setup()
    except Exception as e:
        print(f"⚠ Exemplul 3 a eșuat: {e}")
    
    try:
        example_4_custom_configuration()
    except Exception as e:
        print(f"⚠ Exemplul 4 a eșuat: {e}")
    
    try:
        example_5_inference_pipeline()
    except Exception as e:
        print(f"⚠ Exemplul 5 a eșuat: {e}")
    
    print("\n" + "="*70)
    print("DEMONSTRAȚIE COMPLETĂ!")
    print("="*70)
