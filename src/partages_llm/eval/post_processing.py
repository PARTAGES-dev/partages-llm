import os
import re
import json
import yaml
import shutil
import warnings
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from itertools import chain
from math import copysign
from datetime import datetime

import pandas as pd

from ..utils import format_model_name


def rearrange_result_files_for_zeno(results_dir: str, dest_dir: str):
    # regex pour extraire la tâche et le timestamp
    # TODO: Paths
    pattern = re.compile(r"samples_(.+)_(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.\d+)\.jsonl")

    for model in os.listdir(results_dir):
        model_path = os.path.join(results_dir, model)
        if not os.path.isdir(model_path):
            continue

        for file in os.listdir(model_path):
            match = pattern.match(file)
            if not match:
                continue

            task = match.group(1)  # ex: arc_challenge_chat
            timestamp = match.group(2)  # ex: 2025-06-08T19-45-18.063741

            # créer la hiérarchie task/model
            task_dir = os.path.join(dest_dir, task, model)
            os.makedirs(task_dir, exist_ok=True)

            # chemins source et destination
            src_jsonl = os.path.join(model_path, file)
            src_json = os.path.join(model_path, f"results_{timestamp}.json")

            dst_jsonl = os.path.join(task_dir, file)
            dst_json = os.path.join(task_dir, f"results_{timestamp}.json")

            # copie des fichiers
            shutil.copy2(src_jsonl, dst_jsonl)
            if os.path.exists(src_json):
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
    output_dir: str,
    recursive: bool = False,
    output_name: Optional[str] = None,  # TODO: get rid - return dataframe and have script/user write to disk if they want
    completion_stats_config: Optional[Union[str, Dict[str, Union[List[str], bool]]]] = None,  # TODO: get rid of this
    task_group_ref_path: Optional[str] = None,
    base_model_name_only: bool = False,
    existing_results: Optional[pd.DataFrame] = None,
    verbose: bool = False
) -> type(None):
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
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
        existing_results = {"task": [], "model": []}  # TODO: make this a Bunch; for now, I'm avoiding cross-imports between different lib packages
    for result in all_results:
        if result["task"] in existing_results["task"] and result["model"] in existing_results["model"]:
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
    if verbose:
        print("Dataframe: %d rows\n%s" % (len(df), repr(df)))
        if not task_group_ref_path:
            raise RuntimeError("Completion stats requested but no task grouping reference provided")
        with Path(task_group_ref_path).open() as f:
            task_group_ref = yaml.safe_load(f)
        if completion_stats_config:
            if isinstance(completion_stats_config, str):
                with Path(completion_stats_config).open() as f:
                    completion_stats_config = json.load(f)
            completion_stats_task_groups = completion_stats_config["task_groups"] \
                if completion_stats_config["task_groups"] else list(task_group_ref)
            completion_stats_tasks = list(chain(*[
                [t["task"] for t in task_group_ref[group]] for group in completion_stats_task_groups
            ]))
            task_count = len(completion_stats_tasks)
            df_completion_stats = df[
                df.task.isin(completion_stats_tasks) & df.model.isin(completion_stats_config["models"])
            ]
            for model in df_completion_stats.model.drop_duplicates():
                completed_tasks = df_completion_stats.loc[
                    df_completion_stats.model.str.startswith(model), "task"
                ]
                completed_task_count = len(completed_tasks)
                model_completion_stats_disp = f"{model}: {completed_task_count}/{task_count}"
                if completion_stats_config["print_missing"]:
                    if completed_task_count:
                        missing_tasks = set(completion_stats_tasks) - set(completed_tasks)
                        if missing_tasks:
                            model_completion_stats_disp += "\nMissing: " + ", ".join(missing_tasks)
                print(model_completion_stats_disp, end="\n" * 2)
    if not output_name:
        output_name = "eval_results_" + datetime.now().strftime("%y-%m-%d-%H-%M")
    write_path = output_path / (output_name + ".tsv")
    if isinstance(existing_results, pd.DataFrame):
        df = pd.concat((existing_results, df), ignore_index=True)
    df.to_csv(write_path, index=False, encoding="utf-8", sep="\t")
    if verbose: print("\nOutput written to %s" %  write_path)


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
    # n_samples = i + 1
    names = model_name1, "equal", model_name2
    return dict(zip(names, counts))


def head2head_chart(df: pd.DataFrame):
    pass


def _make_table_row(df_row):
    return f"{df_row['model']} & {df_row['score']:>6.2f}$\\pm${df_row['std']:.1f} & {df_row['metric']}"


def csv2table(
    input_path: Union[Path, str],
    output_path: Union[Path, str],
    sep: str="\t",
):
    df = pd.read_csv(input_path, sep=sep).sort_values(by=["task", "model"])
    if df.score.max() <= 1.:
        # convert scores to percentages
        df['score'] *= 100
        df['std'] *= 100
    if isinstance(output_path, str):
        output_path = Path(output_path)
    with output_path.open("w") as f:
        for task in df.task.unique():
            task_df = df[df.task == task]
            n_shots = task_df.nshots.tolist().pop()
            f.write(f"Task: {task}\t{n_shots}-shot\n")
            for _, row in df[df.task == task].iterrows():
                f.write(_make_table_row(row) + "\n")
            f.write("\n")  # empty line between the tasks

