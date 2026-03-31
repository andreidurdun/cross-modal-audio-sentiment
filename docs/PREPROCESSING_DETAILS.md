# Preprocesarea datelor în pipeline-ul MSP-Podcast

Acest document descrie în detaliu pașii de preprocesare a datelor pentru pipeline-ul de recunoaștere a emoțiilor folosind corpusul MSP-Podcast. Preprocesarea implică atât datele audio, cât și cele textuale (transcrieri în engleză și spaniolă).

## 1. Structura generală a pipeline-ului

- **Date sursă**: Fișiere audio `.wav`, transcrieri `.json` și etichete `.csv`/`.json`.
- **Scop**: Extracția de embeddinguri multimodale (audio, text_en, text_es) și pregătirea label-urilor pentru clasificare sau regresie.
- **Output**: Fișiere `.pt` cu embeddinguri și metadate, gata de folosit la antrenare/testare.

---

## 2. Preprocesarea audio

### a. Încărcarea și procesarea fișierelor audio
- Se folosește `soundfile` pentru citirea fișierelor `.wav`.
- Toate fișierele sunt convertite la mono (dacă nu sunt deja), folosind media canalelor.
- Se resamplează la 16 kHz (sau altă rată specificată).
- Opțional: se aplică augmentare de tip "telephony" (downsampling la 8 kHz și upsampling înapoi la 16 kHz) pentru datele de antrenament.
- Se gestionează erorile de citire: dacă un fișier nu poate fi încărcat, se folosește un vector de zerouri cu lungimea corespunzătoare.

### b. Trunchierea/padding-ul audio
- Toate secvențele audio sunt trunchiate la maximum 5 secunde (96.000 de sample-uri la 16 kHz).
- Dacă secvența este mai scurtă, se face padding cu zerouri.

---

## 3. Preprocesarea textului

### a. Încărcarea transcrierilor
- Transcrierile sunt încărcate din fișiere `.json` (engleză și spaniolă).
- Fiecare transcriere este asociată cu un `file_id` (fără extensie `.wav`).
- Dacă o transcriere lipsește, se folosește un string gol.

### b. Tokenizare și embedding
- Textul brut este tokenizat și procesat de backbone-ul corespunzător (RoBERTa EN/ES).
- Se extrag embeddinguri pentru fiecare secvență, cu padding/trunchiere la 100 de tokeni.

---

## 4. Preprocesarea label-urilor

### a. Clasificare
- Etichetele brute (`EmoClass`) sunt mapate la cele 3 clase finale: `unsatisfied`, `neutral`, `satisfied`.
- Mapping-ul folosește atât clasa de emoție, cât și scorul de valență (`EmoVal`):
    - Emoții negative (`Ang`, `Sad`, `Dis`, `Con`, `Fea`) → `unsatisfied`
    - `Hap` → `satisfied`
    - `Neu` → `neutral`
    - Alte emoții: dacă `EmoVal` ≤ 3.5 → `unsatisfied`, dacă `EmoVal` ≥ 4.5 → `satisfied`, altfel `neutral`.
- Se elimină exemplele fără label valid.

### b. Regresie
- Pentru taskuri de regresie, valorile `EmoVal` (valence) și `EmoAct` (arousal) sunt extrase din fișierele de label și adăugate la embeddinguri.

---

## 5. Salvarea embeddingurilor și metadatelor

- Embeddingurile pentru fiecare modalitate sunt concatenate și salvate ca tensor PyTorch (`.pt`).
- Se salvează și lista de `file_ids`, label-urile, precum și metadate despre dimensiuni, partition, data extragerii etc.
- Pentru regresie, se adaugă tensorii `valence` și `arousal`.

---

## 6. Rezumat flux preprocesare

1. **Încărcare metadate și split** (train/val/test)
2. **Încărcare și procesare audio** (mono, resample, trunchiere/padding)
3. **Încărcare și procesare text** (transcrieri, tokenizare, embedding, padding/trunchiere)
4. **Mapare și filtrare label-uri** (clasificare sau regresie)
5. **Salvare embeddinguri și metadate**

---

## 7. Coduri relevante
- `src/data/dataset.py` — încărcare și filtrare metadate, asociere transcrieri, mapare label-uri
- `src/preprocessing/audio_processor.py` — încărcare și procesare audio
- `scripts/extract_and_save_embeddings.py` — pipeline complet de extragere și salvare embeddinguri

---

## 8. Observații
- Pipeline-ul este robust la lipsa unor fișiere/transcrieri: folosește fallback-uri (zerouri/string gol).
- Padding-ul și trunchierea asigură compatibilitatea batch-urilor la antrenare.
- Pentru taskuri de regresie, este necesară adăugarea valorilor valence/arousal în fișierele de embeddinguri (vezi utilitarul din extract_and_save_embeddings.py).

---

## 9. Statistici seturi de date

Numar total exemple (Train): 169190
Numar total exemple (Val): 34399

Numar exemple per label (Train):
  - unsatisfied: 63693
  - neutral: 65026
  - satisfied: 40471

Numar exemple per label (Val):
  - unsatisfied: 15599
  - neutral: 10459
  - satisfied: 8341
