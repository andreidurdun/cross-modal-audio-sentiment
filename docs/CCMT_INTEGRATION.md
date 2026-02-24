# Integrarea CCMT în Arhitectura Multimodală

## Prezentare Generală

Această integrare combină:
1. **Backbones pretrenate** (RoBERTa-EN, RoBERTa-ES, WavLM) - frozen pentru feature extraction
2. **Fusion Adapters** - convertesc embeddings în patch tokens
3. **CCMT (Cascaded Cross-Modal Transformer)** - realizează fuziunea multimodală și predicția

## Arhitectură

```
Input:
├── Text EN → RoBERTa-EN → (B, 768) ──┐
├── Text ES → RoBERTa-ES → (B, 768) ──┤
└── Audio   → WavLM      → (B, 768) ──┤
                                       │
                    Fusion Adapters    │
                    (patch generation) │
                                       ↓
                    Multimodal Tokens: (B, 300, 1024)
                    [100 patches ES | 100 patches EN | 100 patches Audio]
                                       │
                                       ↓
                    CCMT (Cross-Modal Transformer)
                    ├── Language Cross-Attention (ES ↔ EN)
                    └── Speech Cross-Attention (Text ↔ Audio)
                                       │
                                       ↓
                    Predicție: (B, 3) [unsatisfied, neutral, satisfied]
```

## Componente

### 1. Fusion Adapters (`src/models/fusion_net.py`)

Convertesc embeddings de dimensiuni diferite în token patches uniforme:

```python
from src.models import MultimodalFusionAdapter

adapter = MultimodalFusionAdapter(
    text_en_dim=768,      # RoBERTa output
    text_es_dim=768,      # RoBERTa output
    audio_dim=768,        # WavLM output
    ccmt_dim=1024,        # Dimensiune target pentru CCMT
    num_patches_per_modality=100,  # 100 patches × 3 modalități = 300 total
)

# Input: embeddings separate
# Output: (batch, 300, 1024) patch tokens
tokens = adapter(text_en_emb, text_es_emb, audio_emb)
```

**Caracteristici:**
- `ModalityAdapter`: Adaptor generic pentru text/features simple
- `AudioAdapter`: Adaptor specializat cu temporal pooling pentru audio sequences
- Suport pentru dimensiuni variabile de input
- LayerNorm + GELU pentru stabilitate

### 2. CCMT Layer (`src/models/ccmt_layer.py`)

Transformer cascadat pentru cross-modal attention:

```python
from src.models import CascadedCrossModalTransformer

ccmt = CascadedCrossModalTransformer(
    num_classes=3,
    num_patches=300,      # 100 per modalitate × 3
    dim=1024,             # Dimensiune token
    depth=6,              # 6 transformer layers
    heads=8,              # 8 attention heads
    mlp_dim=2048,         # MLP hidden size
    dropout=0.2,
)

# Input: (batch, 300, 1024) multimodal tokens
# Output: (batch, 3) class predictions cu sigmoid
predictions = ccmt(tokens)
```

**Arhitectură:**
- **Level 1:** Cross-attention între Text ES ↔ Text EN (fuziune lingvistică)
- **Level 2:** Cross-attention între Text fuzionat ↔ Audio (fuziune multimodală)
- **Output:** MLP head cu LayerNorm + Sigmoid

### 3. Full Model (`src/models/full_model.py`)

Arhitectura end-to-end completă:

```python
from src.models import load_full_multimodal_model

model = load_full_multimodal_model(
    text_en_checkpoint="checkpoints/roberta_text_en",
    text_es_checkpoint="checkpoints/roberta_text_es",
    num_classes=3,
    ccmt_dim=1024,
    num_patches_per_modality=100,
    ccmt_depth=6,
    ccmt_heads=8,
    freeze_backbones=True,  # Frozen pentru feature extraction
)

# Predicție cu texte și audio
predictions, classes = model.predict(
    text_en=["I feel great!"],
    text_es=["Me siento genial!"],
    audio_emb=audio_embeddings,
)
```

## Utilizare

### Opțiunea 1: Model Complet (cu backbones)

```python
from src.models import load_full_multimodal_model

# Încarcă model cu backbones pretrenate
model = load_full_multimodal_model(
    text_en_checkpoint="checkpoints/roberta_text_en",
    text_es_checkpoint="checkpoints/roberta_text_es",
    freeze_backbones=True,
)

# Predicție directă cu text
predictions = model(
    text_en=["Sample text"],
    text_es=["Texto de muestra"],
    audio_emb=audio_features,
)
```

### Opțiunea 2: Embeddings Pre-calculate (mai rapid)

```python
# Pre-calculează embeddings cu backbones
text_en_emb = text_en_backbone(texts_en)  # (B, 768)
text_es_emb = text_es_backbone(texts_es)  # (B, 768)
audio_emb = audio_backbone(audio)         # (B, 768)

# Inferență DOAR prin fusion + CCMT (foarte rapid!)
predictions = model(
    text_en_emb=text_en_emb,
    text_es_emb=text_es_emb,
    audio_emb=audio_emb,
)
```

### Opțiunea 3: Construire Manuală

```python
from src.models import (
    MultimodalFusionAdapter,
    CascadedCrossModalTransformer,
    load_text_backbones,
)

# Încarcă backbones
backbones = load_text_backbones(freeze=True)

# Construiește fusion
fusion = MultimodalFusionAdapter(
    text_en_dim=768,
    text_es_dim=768,
    audio_dim=768,
    ccmt_dim=1024,
    num_patches_per_modality=100,
)

# Construiește CCMT
ccmt = CascadedCrossModalTransformer(
    num_classes=3,
    num_patches=300,
    dim=1024,
    depth=6,
    heads=8,
    mlp_dim=2048,
)

# Pipeline manual
text_en_emb = backbones['text_en'](texts_en)
text_es_emb = backbones['text_es'](texts_es)
audio_emb = audio_backbone(audio)

tokens = fusion(text_en_emb, text_es_emb, audio_emb)
predictions = ccmt(tokens)
```

## Training

### Script de Training

```bash
python scripts/train_ccmt_multimodal.py
```

**Configurare:**
- Backbones: **frozen** (pretrenate)
- Trainable: **fusion adapters + CCMT**
- Optimizer: AdamW (lr=1e-4, wd=0.01)
- Loss: BCELoss cu class weights
- Scheduler: ReduceLROnPlateau
- Early stopping: patience=7

### Custom Training Loop

```python
from src.models import load_full_multimodal_model
import torch.nn as nn

# Model
model = load_full_multimodal_model(freeze_backbones=True)

# Optimizer (doar parametrii trainable)
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)

# Loss
criterion = nn.BCELoss()

# Training step
model.train()
predictions = model(text_en_emb=..., text_es_emb=..., audio_emb=...)

# One-hot encode labels
labels_onehot = torch.zeros_like(predictions)
labels_onehot.scatter_(1, labels.unsqueeze(1), 1.0)

loss = criterion(predictions, labels_onehot)
loss.backward()
optimizer.step()
```

## Demo și Teste

### Test Integrare

```bash
python scripts/demo_ccmt_integration.py
```

Rulează 5 exemple demonstrative:
1. Încărcare model și predicție simplă
2. Pre-calculare embeddings pentru eficiență
3. Setup training complet
4. Configurare CCMT personalizată
5. Pipeline complet de inferență

### Test Componente Individual

```bash
# Test fusion adapters
python src/models/fusion_net.py

# Test CCMT
python src/models/ccmt_layer.py

# Test model complet
python src/models/full_model.py
```

## Parametri Trainable

Cu `freeze_backbones=True` (recomandat):

| Component | Parameters | Trainable |
|-----------|-----------|-----------|
| RoBERTa-EN | ~125M | ❌ Frozen |
| RoBERTa-ES | ~125M | ❌ Frozen |
| WavLM | ~95M | ❌ Frozen |
| Fusion Adapters | ~31M | ✅ Yes |
| CCMT | ~25M | ✅ Yes |
| **Total** | ~401M | ~56M (14%) |

**Avantaje:**
- Training rapid (doar 56M parametri)
- Nu necesită backpropagation prin backbones
- Embeddings pot fi pre-calculate
- Memory efficient

## Configurații Recomandate

### Configurație Standard (Acuratețe Maximă)

```python
model = load_full_multimodal_model(
    ccmt_dim=1024,
    num_patches_per_modality=100,
    ccmt_depth=6,
    ccmt_heads=8,
    ccmt_mlp_dim=2048,
    ccmt_dropout=0.2,
)
```

### Configurație Lightweight (Inferență Rapidă)

```python
model = load_full_multimodal_model(
    ccmt_dim=512,
    num_patches_per_modality=50,
    ccmt_depth=4,
    ccmt_heads=4,
    ccmt_mlp_dim=1024,
    ccmt_dropout=0.15,
)
```

### Configurație Heavy (Dataset Mare)

```python
model = load_full_multimodal_model(
    ccmt_dim=1536,
    num_patches_per_modality=150,
    ccmt_depth=8,
    ccmt_heads=12,
    ccmt_mlp_dim=3072,
    ccmt_dropout=0.25,
)
```

## Workflow Recomandat

### 1. Pre-procesare Embeddings (Offline)

```python
# Salvează embeddings pentru întreg dataset-ul
from src.models import load_text_backbones

backbones = load_text_backbones(freeze=True)

for batch in dataset:
    text_en_emb = backbones['text_en'](batch['text_en'])
    text_es_emb = backbones['text_es'](batch['text_es'])
    # Salvează embeddings pe disc
    torch.save({
        'text_en_emb': text_en_emb,
        'text_es_emb': text_es_emb,
        'audio_emb': audio_emb,
        'labels': labels,
    }, f'embeddings/batch_{i}.pt')
```

### 2. Training pe Embeddings Pre-calculate

```python
# Dataset încarcă embeddings direct
class EmbeddingDataset(Dataset):
    def __getitem__(self, idx):
        data = torch.load(f'embeddings/batch_{idx}.pt')
        return data

# Training rapid (fără forward prin backbones)
trainer = CCMTTrainer(model, train_loader, ...)
trainer.train()
```

### 3. Inferență Optimizată

```python
# Pre-calculează embeddings o singură dată
text_en_emb = cache_or_compute(text_en)
text_es_emb = cache_or_compute(text_es)

# Inferență instantanee
predictions = model(
    text_en_emb=text_en_emb,
    text_es_emb=text_es_emb,
    audio_emb=audio_emb,
)
```

## Troubleshooting

### OOM (Out of Memory)

1. Reduce `batch_size`
2. Reduce `num_patches_per_modality` (e.g., 50 în loc de 100)
3. Reduce `ccmt_dim` (e.g., 512 în loc de 1024)
4. Reduce `ccmt_depth` (e.g., 4 în loc de 6)

### Training Instabil

1. Reduce `learning_rate` (e.g., 5e-5 în loc de 1e-4)
2. Crește `weight_decay` (e.g., 0.05)
3. Adaugă gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`
4. Reduce `ccmt_dropout`

### Underfitting

1. Crește `ccmt_depth` (mai multe layers)
2. Crește `ccmt_heads` (mai multe attention heads)
3. Crește `num_patches_per_modality` (mai multă capacitate)
4. Reduce `weight_decay`
5. Reduce `dropout`

## Fișiere Generate

După training:

```
checkpoints/ccmt_multimodal/
├── best_model.pt              # Best model pe validation
├── checkpoint_epoch_5.pt      # Checkpoint intermediar
├── checkpoint_epoch_10.pt
└── training_history.json      # Metrics history
```

## Next Steps

1. **Integrare Audio Backbone:** Înlocuiește `audio_emb` dummy cu WavLM real
2. **Hyperparameter Tuning:** Optimizează configurația CCMT
3. **Data Augmentation:** Adaugă augmentări specifice la training
4. **Ensemble:** Combină predicții de la multiple modele CCMT
5. **Distillation:** Distilează modelul mare într-unul mai mic pentru deployment

## Referințe

- RoBERTa: Liu et al., 2019
- WavLM: Chen et al., 2022
- Cross-Modal Transformers: Tsai et al., 2019
- Cascaded Attention: Zhang et al., 2021
