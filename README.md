# partages-llm

> Ce référentiel contient le code utilisé pour les expériences sur des modèles de langue décodeur dans le cadre de la première phase du projet PARTAGES.

## Overview

Le projet PARTAGES vise à développer des modèles de langues spécialisés destinés à être utilisés dans l'automatisation des tâches de traitement de documents dans le système de santé français, tout en mettant les ressources associées (modèles, code, données) à disposition publiquement.
Ces travaux se focalisaient sur les modèles décodeurs génératifs, dans le cadre des objectifs de spécialiser des modèles généralistes génératifs de type causaux sur le domaine médical en français et sur leur validation à l'aide de benchmarks publics.
Dans ce cadre, nous avons testé le pré-entraînement continu ainsi que des expériences à petite échelle sur l'affinage supervisé en utilisant les corpus et les ensembles de données textuelles recueillis au sein de PARTAGES.
Nos résultats remettent en question l'utilité de faire ce genre d'adaptation aux domaines spécialisés dans à l'ère des grands modèles généralistes puissants.

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

- `clm-corpus-processing` - paramètres du train/test/split du corpus `PARCOMED`
- `fsdp` : configuration du framework Fully-Sharded Data Parallel pour l'entraînement des différents modèles de fondation
- `lm-eval` : regroupements thématiques des tâches d'évaluation
- `templates/mcqa-processing` : configuration des instructions pour les expériences avec MEDIQAL et FrenchMedMCQA
- `train` : exemples de configuration globale des scripts d'entraînement

### `src/`

Description of the main library modules and what they provide.

### `scripts/`

#### `eda/`
`topic_modelling.py` contient le code d'implémentation utilisé pour des analyses exploratoires sur les sujets médicaux dans le corpus.

#### `eval/`
Le script `do-eval.sh` gère le benchmarking avec `lm-evaluation-harness`, et `mcq_inference.py` l'évaluation des modèles sur la tâche MCQ de la partie étiquetée du corpus.

#### `postprocess/`
Outil pour faciliter le traitement des sorties des évaluations de `lm-evaluation-harness`.

#### `preprocess/`
Chaîne de traitements `PARCOMED` :
1. `clean_clm_dataset.py`
2. `deduplicate_clm_dataset.py` (facultatif)
3. `make_clm_dataset_mix.py` (facultatif)
4. `prepare_clm_tokens.py` (tokenisation pour un modèle précisé)

#### `train/`
- `clm` (Causal Language Modelling)
- `sft` (Supervised Fine-Tuning)
- `sts` (Semantic Textual Similarity)


