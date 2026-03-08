# Multimodal Sentiment Project

Three-channel pipeline for MSP-Podcast 2.0:

1. **Speech → Text**: Whisper generates transcripts saved under `data/transcripts/`.
2. **Text sentiment**: English and Romanian experts plus translation cache captured in `src/models/text_ensemble.py`.
3. **Audio sentiment**: WavLM + CCMT fusion network living in `src/models/fusion_net.py`.

## Repository Layout

```
configs/
  model_config.json      # backbone- and head-specific settings

data/
  raw_audio/             # MSP .wav files (gitignored)
  metadata/              # CSV / partition metadata
  transcripts/           # Whisper outputs (JSON)
  cache/                 # translation caches, pickles
  processed/             # curated CSV splits

checkpoints/
  audio_adapter/
  text_en_adapter/
  text_ro_adapter/

src/
  preprocessing/         # audio, transcription, translation utilities
  data/                  # PyTorch datasets + collate
  models/                # backbones, CCMT fusion, text ensembles
  utils/                 # metrics, helpers, logging

scripts/
  01_run_transcription.py
  02_run_translation.py
  train.py

notebooks/
  01_explore_data.ipynb
  02_test_whisper.ipynb
  03_analyze_results.ipynb
```

## Getting Started

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

1. **Transcribe audio**
   ```bash
   python scripts/01_run_transcription.py --audio-dir data/raw_audio
   ```
2. **Translate transcripts**
   ```bash
   python scripts/02_run_translation.py --input data/transcripts/transcripts_en.json
   ```
3. **Train multimodal model**
   ```bash
  python scripts/train.py
   ```

Both `data/` and `checkpoints/` are ignored by git; copy sample metadata if you need to share minimal repros.
