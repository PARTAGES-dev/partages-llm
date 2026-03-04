# partages-llm

> Ce référentiel contient le code utilisé pour les expériences sur des modèles de langue décodeur dans le cadre de la première phase du projet PARTAGES.

## Overview

(report abstract)

## Installation

```bash
# e.g. clone and install
git clone https://github.com/PARTAGES-dev/partages-llm.git
cd partages-llm
pip install -e .
```

Certaines dépendances supplémentaires doivent être installées pour pouvoir utiliser pleinement le code présenté ici :
- Pour l'évaluation, [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness)
- Pour la déduplication du corpus CLM, comme implémenté dans `scripts/preprocess/deduplicate_clm_dataset.py`, [Onion](https://corpus.tools/wiki/Onion)

## Project Structure

```
partages-llm/
├── configs/  # Configuration files for parameterising pipeline components
|   ├── clm-corpus-processing
|   ├── fsdp
|   ├── lm-eval
|   ├── merge
|   ├── sft-hps
|   ├── templates
|   └── train
├── scripts/  # Runnable recipes for experiments and data processing
|   ├── eda
|   ├── eval
|   ├── postprocess
|   ├── preprocess
|   └── train
|       ├── clm  # Causal Language Modelling
|       ├── sft  # Supervised Fine-Tuning
|       └── sts  # Semantic Textual Similarity
└── src/partages-llm  # Core library code
```

### `configs/`

Description of what lives here — model configs, training hyperparameters, dataset settings, etc.

### `src/`

Description of the main library modules and what they provide.

### `scripts/`

#### `eda/`
#### `eval/`
Le script `do-eval.sh` gère le benchmarking avec `lm-evaluation-harness`, et `mcq_inference.py` l'évaluation des modèles sur la tâche MCQ de la partie étiquetée du corpus.

#### `postprocess/`
#### `preprocess/`
#### `train/`


## Usage

### Reproducing Experiments

```bash
# example command
python scripts/train/clm/run_clm_trainer.py --config configs/train/your_config.yaml
```

## Data

liens aux repos, description basique du prep 

## Results

cf rapport technique...