# Ghid: Salvare și Utilizare Embeddings Pre-calculate

## Prezentare

Aceste scripturi permit **pre-calcularea și salvarea embedding-urilor** de la backbones (RoBERTa-EN, RoBERTa-ES, WavLM), astfel încât training-ul CCMT să fie **mult mai rapid**.

### Avantaje:
✅ **Training 5-10x mai rapid** - nu mai calculezi embeddings la fiecare epocă  
✅ **Economie GPU memory** - backbones nu mai sunt în memorie  
✅ **Refolosire** - embeddings calculate o singură dată, folosite pentru multe experimente  
✅ **Debugging ușor** - poți testa CCMT independent de backbones

---

## 1. Extragere Embeddings

### Script: `extract_and_save_embeddings.py`

#### 1.1 Extrage pentru o partiție specifică

```bash
# Doar training set
python scripts/extract_and_save_embeddings.py --partition train

# Doar validation set
python scripts/extract_and_save_embeddings.py --partition val

# Doar test set
python scripts/extract_and_save_embeddings.py --partition test1
```

#### 1.2 Extrage pentru toate partițiile

```bash
# Extrage train, val, test1
python scripts/extract_and_save_embeddings.py --partition all
```

#### 1.3 Opțiuni avansate

```bash
# Cu proiecție 128-dim (mai mic, mai rapid)
python scripts/extract_and_save_embeddings.py --partition all --projection-dim 128

# Batch size mai mare (mai rapid pe GPU puternice)
python scripts/extract_and_save_embeddings.py --partition all --batch-size 64

# Salvează și samples individuali (pentru debugging)
python scripts/extract_and_save_embeddings.py --partition train --save-individually

# Fără proiecție (dimensiuni native: 768)
python scripts/extract_and_save_embeddings.py --partition all --projection-dim 0

# Output personalizat
python scripts/extract_and_save_embeddings.py --partition all --output-dir my_embeddings
```

### Output:

După rulare, vei avea:

```
data/embeddings/
├── embeddings_train.pt       # Embeddings train (toate în unul)
├── embeddings_val.pt         # Embeddings validation
├── embeddings_test1.pt       # Embeddings test
├── metadata_train.json       # Metadata
├── metadata_val.json
├── metadata_test1.json
├── extraction_summary.json   # Rezumat extragere
└── individual/               # (opțional) samples individuali
    ├── train/
    │   ├── sample_001.pt
    │   └── ...
    └── ...
```

### Structură fișier embeddings:

```python
embeddings_train.pt:
{
    'text_en': Tensor [N, 256],      # N samples, 256-dim
    'text_es': Tensor [N, 256],
    'audio': Tensor [N, 256],
    'labels': Tensor [N],
    'file_ids': List[str],
    'metadata': {
        'partition': 'train',
        'num_samples': N,
        'projection_dim': 256,
        'embedding_dims': {...},
        'extraction_date': '...',
    }
}
```

---

## 2. Utilizare în Training

### Script: `precomputed_embeddings_dataset.py`

#### 2.1 Quick Start: Load Dataset

```python
from scripts.precomputed_embeddings_dataset import PrecomputedEmbeddingsDataset
from torch.utils.data import DataLoader

# Load dataset
dataset = PrecomputedEmbeddingsDataset(
    embeddings_dir="data/embeddings",
    partition="train",
)

# Create DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Iterate
for batch in loader:
    text_en_emb = batch['text_en_emb']  # (batch_size, 256)
    text_es_emb = batch['text_es_emb']  # (batch_size, 256)
    audio_emb = batch['audio_emb']      # (batch_size, 256)
    labels = batch['label']             # (batch_size,)
```

#### 2.2 Training Loop Complet

```python
from scripts.precomputed_embeddings_dataset import create_dataloaders
from src.models import load_full_multimodal_model
import torch.nn as nn

# 1. Create DataLoaders
dataloaders = create_dataloaders(
    embeddings_dir="data/embeddings",
    batch_size=32,
)

train_loader = dataloaders['train']
val_loader = dataloaders['val']

# 2. Load model (NU încărcăm backbones!)
# Modelul primește direct embeddings
model = load_full_multimodal_model(
    freeze_backbones=True,
    projection_dim=256,
)

device = "cuda"
model = model.to(device)

# 3. Optimizer (doar fusion + CCMT)
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4,
)

criterion = nn.CrossEntropyLoss()

# 4. Training Loop
for epoch in range(20):
    model.train()
    
    for batch in train_loader:
        # Move to device
        text_en_emb = batch['text_en_emb'].to(device)
        text_es_emb = batch['text_es_emb'].to(device)
        audio_emb = batch['audio_emb'].to(device)
        labels = batch['label'].to(device)
        
        # Forward (direct cu embeddings!)
        predictions = model(
            text_en_emb=text_en_emb,
            text_es_emb=text_es_emb,
            audio_emb=audio_emb,
        )
        
        # Loss & backward
        loss = criterion(predictions, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

#### 2.3 Test Scripts

```bash
# Test dataset loading
python scripts/precomputed_embeddings_dataset.py --example usage

# Test dataloaders creation
python scripts/precomputed_embeddings_dataset.py --example dataloaders
```

---

## 3. Workflow Complet

### Step 1: Antrenează Backbones (dacă nu sunt deja)

```bash
# Text EN
python scripts/train_roberta_text_en.py

# Text ES
python scripts/train_roberta_text_es.py

# Audio
python scripts/train_wavlm_audio.py
```

### Step 2: Extrage Embeddings

```bash
# Extrage pentru toate partițiile
python scripts/extract_and_save_embeddings.py --partition all --batch-size 32
```

**Timp estimat:** ~10-30 minute pe GPU (depinde de dataset size)

### Step 3: Training CCMT cu Embeddings Pre-calculate

```bash
# Modifică train_ccmt_multimodal.py să folosească PrecomputedEmbeddingsDataset
# sau creează un script nou bazat pe exemplele de mai sus
```

---

## 4. Comparație: Cu vs Fără Pre-calculare

### Fără Pre-calculare (Slow):

```
Epocă:
├── Încarcă audio raw + text
├── Forward RoBERTa-EN (125M params)
├── Forward RoBERTa-ES (125M params)
├── Forward WavLM (95M params)
├── Forward Fusion + CCMT
└── Backward Fusion + CCMT

Timp per epocă: ~15-20 min
Memory GPU: ~18 GB
```

### Cu Pre-calculare (Fast):

```
Pre-processing (o singură dată):
└── Extrage toate embeddings → Salvează pe disc

Training (de fiecare epocă):
├── Încarcă embeddings de pe disc
├── Forward Fusion + CCMT
└── Backward Fusion + CCMT

Timp per epocă: ~2-3 min
Memory GPU: ~6 GB
```

**Speedup: 5-10x mai rapid! 🚀**

---

## 5. Tips & Best Practices

### ✅ DO:
- Folosește `projection_dim=256` pentru dimensiune uniformă
- Pre-calculează embeddings pentru toate partițiile
- Salvează embeddings cu același `projection_dim` ca model-ul
- Verifică că embeddings există înainte de training

### ❌ DON'T:
- Nu șterge embeddings după training (refolosește-le!)
- Nu folosi dimensiuni diferite între extragere și model
- Nu încarca backbones în memorie dacă ai embeddings pre-calculate

### 💡 Recommendations:
- **Pentru experimente rapide:** projection_dim=128
- **Pentru acuratețe maximă:** projection_dim=256 sau 512
- **Pentru debugging:** Salvează `--save-individually`

---

## 6. Troubleshooting

### Eroare: "Embeddings file not found"

**Soluție:** Rulează mai întâi scriptul de extragere:
```bash
python scripts/extract_and_save_embeddings.py --partition train
```

### Eroare: "cuda out of memory" la extragere

**Soluții:**
```bash
# Reduce batch size
python scripts/extract_and_save_embeddings.py --partition all --batch-size 16

# Sau folosește CPU (mai lent)
python scripts/extract_and_save_embeddings.py --partition all --device cpu
```

### Eroare: Dimensiuni incompatibile

**Cauză:** projection_dim diferit între extragere și model

**Soluție:** Folosește același projection_dim:
```bash
# Extragere
python scripts/extract_and_save_embeddings.py --projection-dim 256

# Training
model = load_full_multimodal_model(projection_dim=256)
```

---

## 7. File Sizes

### Estimări pentru MSP-Podcast:

| Partition | Samples | projection_dim=256 | projection_dim=768 |
|-----------|---------|-------------------|-------------------|
| train | ~8,000 | ~50 MB | ~150 MB |
| val | ~1,500 | ~10 MB | ~30 MB |
| test1 | ~1,500 | ~10 MB | ~30 MB |
| **TOTAL** | ~11,000 | **~70 MB** | **~210 MB** |

💾 Foarte mic comparativ cu beneficiile!

---

## 8. Advanced: Custom Embeddings

### Extract doar pentru anumite samples:

```python
from scripts.extract_and_save_embeddings import EmbeddingExtractor

extractor = EmbeddingExtractor(
    output_dir="custom_embeddings",
    projection_dim=256,
)

# Custom logic
dataset = MSP_Podcast_Dataset(partition='train')
filtered_indices = [i for i in range(100)]  # Primele 100

# Extract și salvează manual
# ...
```

### Load embeddings custom:

```python
import torch

embeddings = torch.load("data/embeddings/embeddings_train.pt")

# Access componente
text_en = embeddings['text_en']  # Tensor
labels = embeddings['labels']    # Tensor
file_ids = embeddings['file_ids']  # List
```

---

## Summary

✅ **extract_and_save_embeddings.py** - Extrage embeddings de la backbones  
✅ **precomputed_embeddings_dataset.py** - Dataset loader pentru embeddings  
✅ **Training 5-10x mai rapid**  
✅ **~70 MB storage** pentru întreg dataset  
✅ **Reutilizabil** pentru multiple experimente

**Next Steps:**
1. Extrage embeddings: `python scripts/extract_and_save_embeddings.py --partition all`
2. Test loading: `python scripts/precomputed_embeddings_dataset.py --example usage`
3. Training CCMT cu embeddings pre-calculate

🚀 **Happy Fast Training!**
