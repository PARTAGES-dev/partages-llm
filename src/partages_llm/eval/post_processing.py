import re
import json
import shutil
import warnings
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from math import copysign

import pandas as pd

from ..utils import Bunch, format_model_name

FILENAME_PATTERN = re.compile(r"samples_(.+)_(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.\d+)\.jsonl")  # extraire la tâche et le timestamp


def rearrange_result_files_for_zeno(results_dir: Path, dest_dir: Path):
    for model_path in results_dir.iterdir():
        if not model_path.is_dir():
            continue

        for fp in model_path.iterdir():
            match = FILENAME_PATTERN.match(fp.name)
            if not match:
                continue

            task = match.group(1)  # ex: arc_challenge_chat
            timestamp = match.group(2)  # ex: 2025-06-08T19-45-18.063741

            # créer la hiérarchie task/model
            task_dir = dest_dir / task / model_path.name
            task_dir.mkdir(parents=True, exist_ok=True)

            # chemins source et destination
            src_jsonl = model_path / fp.name
            src_json = model_path / f"results_{timestamp}.json"
            dst_jsonl = task_dir / fp.name
            dst_json = task_dir / f"results_{timestamp}.json"

            # copie des fichiers
            shutil.copy2(src_jsonl, dst_jsonl)
            if src_json.exists():
                shutil.copy2(src_json, dst_json)
            else:
                warnings.warn(f"Missing JSON file for {src_jsonl}", RuntimeWarning)


def get_task_results(
    filepath: Path,
    model_name: str,
    input_path: Path,
    base_model_name_only: bool = False,
    verbose: bool = False
) -> List[Dict[str, Any]]:
    """Read a results*.json file & return the results in a dict"""
    if verbose: print("Reading file: %s" % filepath.name)
    with filepath.open("r", encoding="utf-8") as f:
        content = json.load(f)
    if verbose: print("Content loaded: %d task(s)" % len(content["results"]))
    num_few_shots = 0
    for task_config in content["configs"].values():
        num_few_shots = task_config.get("num_fewshot")
    results = content.get("results", {})
    all_entries = []
    for task, result_values in results.items():
        metrics = {}
        for key, value in result_values.items():
            if "," in key and not key.startswith("acc_norm"):
                metric_name, suffix = key.split(",", 1)
                std_key = f"{metric_name}_stderr,{suffix}"
                std_value = result_values.get(std_key, "NA")
                if isinstance(value, dict):
                    if "Value" in value.keys():
                        metrics["spearman_corr"] = (value["Value"], value["pvalue"])
                else:
                    metrics[metric_name] = (value, std_value)
        for metric, (score, std) in metrics.items():
            entry = {
                "task": task,
                "model": format_model_name(model_name, base_model_name_only),
                "metrics+agg": [(metric, "none")],  # Pas d'agg explicite connue
                "scores": [(score, std)],
                "nshots": num_few_shots
            }
            all_entries.append(entry)
    if verbose: print("Entries gathered: %d" % len(all_entries))
    return all_entries


def gather_lm_eval_results_by_domain(
    input_dir: Union[str, List[str]],
    recursive: bool = False,
    base_model_name_only: bool = False,
    existing_results: Optional[pd.DataFrame] = None,
    verbose: bool = False
) -> pd.DataFrame:
    all_results = []
    if isinstance(input_dir, str):
        input_dir = [input_dir]
    for input_str in input_dir:
        input_path = Path(input_str)
        for run_path in input_path.rglob("*/") if recursive else input_path.glob("*/"):
            for i, filepath in enumerate(run_path.glob("results*.json")):
                if verbose and not i: print("Looking @ %s" % run_path.name)
                all_results.extend(
                    get_task_results(
                        filepath, run_path.name, input_path, base_model_name_only
                    )
                )
    if verbose: print("All results gathered: %d" % len(all_results))
    rows = []
    if existing_results is None:
        existing_results = Bunch(task=[], model=[])
    for result in all_results:
        if result["task"] in existing_results.task and result["model"] in existing_results.model:
            continue
        metrics = result["metrics+agg"]
        scores = result["scores"]
        for (metric, agg), (score, std) in zip(metrics, scores):
            if "stderr" not in metric:
                rows.append({
                    "task": result["task"],
                    "model": result["model"],
                    "metric": metric,
                    "aggregation": agg,
                    "score": score,
                    "std": std,
                    "nshots": result["nshots"]
                })
    if verbose: print("Formatted: %d rows" % len(rows))
    df = pd.DataFrame(rows).drop_duplicates()
    if verbose: print("Dataframe: %d rows\n%s" % (len(df), repr(df)))
    if isinstance(existing_results, pd.DataFrame):
        df = pd.concat((existing_results, df), ignore_index=True)
    return df
    


def get_lm_eval_task_outputs_by_model(
    results_dir: Union[Path, str],
    task: str
) -> Dict[str, List[Dict[str, Any]]]:
    if not isinstance(results_dir, Path):
        results_dir = Path(results_dir)
    outputs = {}
    for model_dir in results_dir.iterdir():
        model_name = format_model_name(model_dir.name)
        samples = []
        for fp in model_dir.glob(f"samples_{task}*.jsonl"):
            with fp.open() as f:
                samples.extend([json.loads(line) for line in f])
        outputs[model_name] = samples
    return outputs


def compare_head2head(
    outputs: Dict[str, List[Dict[str, Any]]],
    model_name1: str,
    model_name2: str,
    metric: str = "acc",
) -> Dict[str, int]:
    counts = [0] * 3
    for i, (sample1, sample2) in enumerate(zip(
        outputs[model_name1], outputs[model_name2]
    )):
        d = sample2["acc"] - sample1["acc"]
        idx = int(copysign(1, d)) + 1 if d else 1
        counts[idx] += 1
    names = model_name1, "equal", model_name2
    return dict(zip(names, counts))


def head2head_chart(df: pd.DataFrame):
    raise NotImplementedError("head2head_chart: WIP")


