# Model Testing Scripts

Acest set de scripturi permite testarea și evaluarea fiecărui model antrenat pe datele de validare, cu salvarea scorurilor și generarea graficelor.

## Scripturi disponibile

### Scripturi individuale de testare

1. **`test_roberta_text_en.py`** - RoBERTa Text EN
   ```bash
   python scripts/test_roberta_text_en.py
   ```
   - Testează modelul RoBERTa antrenat pe texturi în limba engleză
   - Salvează rezultate în: `results/roberta_text_en/`

2. **`test_roberta_text_es.py`** - RoBERTa Text ES
   ```bash
   python scripts/test_roberta_text_es.py
   ```
   - Testează modelul RoBERTa antrenat pe texturi în limba spaniolă
   - Salvează rezultate în: `results/roberta_text_es/`

3. **`test_wavlm_audio.py`** - WavLM Audio
   ```bash
   python scripts/test_wavlm_audio.py
   ```
   - Testează modelul WavLM antrenat pe date audio
   - Salvează rezultate în: `results/wavlm_audio/`

4. **`test_ccmt_multimodal.py`** - CCMT Multimodal
   ```bash
   python scripts/test_ccmt_multimodal.py
   ```
   - Testează modelul CCMT care combină toate modalitățile
   - Salvează rezultate în: `results/ccmt_multimodal/`

### Script master

**`run_all_tests.py`** - Rulează testarea pentru toate modelele simultan
```bash
python scripts/run_all_tests.py
```
- Apelează fiecare script de testare
- Generează raport comparat
- Creează grafice comparative
- Afișează rankinguri după F1 Macro

## Output-uri generate

### Pentru fiecare model:

**Fișierele salvate în `results/{model_name}/`:**

1. **`test_results.json`** - Metrici detaliate
   ```json
   {
     "accuracy": 0.XXXX,
     "f1_macro": 0.XXXX,
     "f1_weighted": 0.XXXX,
     "precision_macro": 0.XXXX,
     "recall_macro": 0.XXXX,
     "f1_per_class": {
       "unsatisfied": 0.XXXX,
       "neutral": 0.XXXX,
       "satisfied": 0.XXXX
     },
     "confusion_matrix": [...],
     "avg_loss": 0.XXXX,
     "num_samples": XXXX
   }
   ```

2. **`confusion_matrix.png`** - Harta de confuzie
   - Vizualizează predicții vs etichete reale

3. **`metrics.png`** - Graficele metricilor principale
   - F1 per clasă
   - Metrici generale (Accuracy, F1, Precision, Recall)
   - Confuzion matrix normalizată
   - Informații rezumat

### De la script-ul master:

**În directorul `results/`:**

1. **`comparison_report.json`** - Raport comparat
   - Metrici ale tuturor modelelor
   - Rankinguri pe bază de diferite metrici

2. **`comparison_plots.png`** - Grafice comparative
   - Comparație Accuracy
   - Comparație F1 scores
   - Scatter Precision vs Recall
   - Comparație toate metricile

3. **`f1_per_class_comparison.png`** - Comparație F1 per clasă
   - F1 pentru fiecare clasă across toate modelele

## Metrici calculate

Pentru fiecare model, sunt calculate:

- **Accuracy**: Procentul de predicții corecte
- **F1 Macro**: Media F1 scores pe toate clasele (igual ponderare)
- **F1 Weighted**: Media F1 scores ponderat cu numărul de eșantioane pe clasă
- **Precision Macro**: Media precision pe toate clasele
- **Recall Macro**: Media recall pe toate clasele
- **F1 per Class**: F1 score pentru fiecare clasă în parte (unsatisfied, neutral, satisfied)
- **Confusion Matrix**: Matrice de predicții vs adevărate etichete

## Exemplu de utilizare

```bash
# Testează un model specific
python scripts/test_roberta_text_en.py

# Testează toate modelele și generează rapoarte
python scripts/run_all_tests.py
```

## Cerințe

- PyTorch
- Transformers
- scikit-learn
- matplotlib
- seaborn
- tqdm

Instalare:
```bash
pip install -r requirements.txt
```

## Notă importantă

- Asigurați-vă că checkpoint-urile modelelor sunt disponibile în directoarele `checkpoints/`
- Datele de validare vor fi auto-încărcate din `MSP_Podcast/`
- Rezultatele vor fi create în directorul `results/` dacă nu există
