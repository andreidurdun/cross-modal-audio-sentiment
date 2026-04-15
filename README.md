## Structura proiectului

### fisiere si directoare

- `configs/training_config.json`: configuratia pentru hiperparametri, separata pe modele si task-uri.
- `src/`: implementarea de baza a modelelor, dataset-urilor, utilitarelor si etapelor de preprocesare.
- `scripts/`: scripturi pentru antrenare, testare, extragere de embeddings, transcriere si traducere.
- `MSP_Podcast/`: datele de lucru, transcriptiile, label-urile si embeddings-urile salvate.
- `checkpoints/`: modele antrenate si checkpoint-uri intermediare.
- `results/`: rezultate de evaluare, metrici.

## Codul din `src`

### `src/data`

- `dataset.py`: incarca dataset-ul MSP-Podcast si unifica accesul la audio, label-uri si transcriptii.
- `text_datasets.py`: dataset-uri pregatite pentru modelele de text, inclusiv tokenizare si ambalare pentru clasificare/regresie.
- `audio_datasets.py`: dataset si colator pentru modelele audio bazate pe WavLM.
- `precomputed_embeddings_dataset.py`: dataset pentru antrenarea modelelor multimodale direct din embeddings precompute.

### `src/models`

- `backbones.py`: incarcarea si configurarea backbone-urilor pentru text si audio.
- `ccmt_layer.py`: componentele de baza ale stratului CCMT folosit la fuziune multimodala.
- `fusion_net.py`: logica de fuziune dintre modalitati si constructia tokenilor multimodali.
- `full_model.py`: definirea modelelor complete, plus functii de incarcare pentru backbone-uri si modele multimodale.

### `src/preprocessing`

- `audio_processor.py`: pregatire si procesare a semnalului audio.
- `transcriber.py`: utilitare pentru transcriere automata a fisierelor audio.
- `translator.py`: utilitare pentru traducerea transcriptiilor in alte limbi.

### `src/utils`

- `config.py`: citirea configuratiilor JSON si accesul la sectiuni de configurare.
- `helpers.py`: utilitare comune, de exemplu pentru seed si reproducibilitate.
- `metrics.py`: metrici de clasificare folosite la evaluare.
- `peft_audio.py`: utilitare pentru integrarea/adaptarea PEFT pe partea audio.
- `regression_trainers.py`: logica reutilizabila pentru antrenarea modelelor de regresie.
- `regression_testers.py`: logica reutilizabila pentru evaluarea modelelor de regresie.

## Scripturile din `scripts`

- `extract_and_save_embeddings.py`: extrage embeddings din backbone-uri si le salveaza pentru antrenare ulterioara.
- `plot_confusion_matrices.py`: genereaza matrice de confuzie si comparatii pentru modelele evaluate.
- `precomputed_embeddings_dataset.py`: utilitar pentru lucrul cu embeddings deja generate.

### transcriere si traducere

- `01_run_transcription.py`: genereaza transcriptii pornind de la fisierele audio.
- `02_run_translation_de.py`: traduce transcriptiile in germana.
- `02_run_translation_es.py`: traduce transcriptiile in spaniola.
- `02_run_translation_fr.py`: traduce transcriptiile in franceza.

### scripturi de antrenare

- `train_roberta_text_en.py`: antreneaza modelul de clasificare pe transcriptii in engleza.
- `train_roberta_text_es.py`: antreneaza modelul de clasificare pe transcriptii in spaniola.
- `train_roberta_text_de.py`: antreneaza modelul de clasificare pe transcriptii in germana.
- `train_roberta_text_fr.py`: antreneaza modelul de clasificare pe transcriptii in franceza.
- `train_wavlm_audio.py`: antreneaza modelul audio pentru clasificare.
- `train_ccmt_classification.py`: antreneaza modelul multimodal CCMT pentru clasificare.

- `train_roberta_text_en_regression.py`: antreneaza varianta de regresie pentru transcriptii in engleza.
- `train_roberta_text_es_regression.py`: antreneaza varianta de regresie pentru transcriptii in spaniola.
- `train_roberta_text_de_regression.py`: antreneaza varianta de regresie pentru transcriptii in germana.
- `train_roberta_text_fr_regression.py`: antreneaza varianta de regresie pentru transcriptii in franceza.
- `train_wavlm_audio_regression.py`: antreneaza modelul audio pentru regresie.
- `train_ccmt_regression.py`: antreneaza modelul multimodal CCMT pentru regresie.

### scripturi de testare

- `test_roberta_text_en.py`: evalueaza modelul de clasificare pe text in engleza.
- `test_roberta_text_es.py`: evalueaza modelul de clasificare pe text in spaniola.
- `test_wavlm_audio.py`: evalueaza modelul de clasificare audio.
- `test_ccmt_multimodal.py`: evalueaza modelul multimodal CCMT pentru clasificare.

- `test_roberta_text_en_regression.py`: evalueaza modelul de regresie pe text englez.
- `test_roberta_text_es_regression.py`: evalueaza modelul de regresie pe text spaniol.
- `test_roberta_text_de_regression.py`: evalueaza modelul de regresie pe text german.
- `test_roberta_text_fr_regression.py`: evalueaza modelul de regresie pe text francez.
- `test_wavlm_audio_regression.py`: evalueaza modelul audio pentru regresie.

## Date, checkpoint-uri si rezultate

- `MSP_Podcast/Labels/`: label-urile emotionale folosite la antrenare si evaluare.
- `MSP_Podcast/Transcription_*.json`: transcriptii pe limbi.
- `MSP_Podcast/Audios/`: fisierele audio brute.
- `MSP_Podcast/embeddings*/`: embeddings salvate pentru diverse combinatii de modalitati.
- `checkpoints/`: checkpoint-uri pentru backbone-uri si modele multimodale.
- `results/`: fisiere JSON, rapoarte si rezultate agregate.
