# Project Guidelines

## Code Style
- Keep changes focused and minimal; do not refactor unrelated modules in the same patch.
- Follow existing Python style in this repository (type hints where already used, Path-based paths, explicit imports).
- Prefer reusable logic in `src/` and keep `scripts/` as thin orchestration entry points.
- Preserve existing Romanian/English mixed comments and strings unless a task explicitly asks for language cleanup.

## Architecture
- Core implementation lives in `src/`:
  - `src/models/`: multimodal model components (`backbones.py`, `fusion_net.py`, `ccmt_layer.py`, `full_model.py`)
  - `src/data/`: dataset loading and label/transcript handling
  - `src/preprocessing/`: audio and text preprocessing utilities
  - `src/utils/`: shared helpers
- Training and evaluation entry points are in `scripts/` (`train_*.py`, `test_*.py`, embedding/transcription/translation scripts).
- Configuration is centralized in `configs/model_config.json`.
- Artifacts are expected under `checkpoints/` and `results/`.

## Build And Test
- Create and activate environment (PowerShell):
  - `python -m venv .venv`
  - `.\.venv\Scripts\Activate.ps1`
  - `pip install -r requirements.txt`
- Typical workflow order:
  - `python scripts/extract_and_save_embeddings.py`
  - `python scripts/train_ccmt_classification.py` (or another `train_*.py`)
  - `python scripts/test_ccmt_multimodal.py` (or matching `test_*.py`)
- Additional utilities:
  - `python scripts/plot_confusion_matrices.py`

## Conventions
- Prefer precomputed embeddings (`MSP_Podcast/embeddings/*.pt`) for CCMT training via `scripts/precomputed_embeddings_dataset.py`.
- Keep dataset partition naming consistent with current code (`train`, `val`, `test1`).
- When adding new training scripts, keep naming aligned with existing pattern: `train_<model_or_task>.py` and `test_<model_or_task>.py`.
- For multimodal classification metrics, report at least macro F1 and accuracy to stay comparable with current validation reports.
- If touching LoRA orthogonalization logic, avoid large `D x D` Gram matrices on flattened updates (`B^T B`); use smaller `K x K` form (`B B^T`) to prevent GPU OOM.

## Project Docs
- Preprocessing details and label mapping: `docs/PREPROCESSING_DETAILS.md`
- Validation baseline summary: `results/rezultate_validare.md`
