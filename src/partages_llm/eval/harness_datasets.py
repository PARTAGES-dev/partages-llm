"""Builder functions for lm-evaluation-harness configuration
"""
import json
from pathlib import Path
from typing import List
from collections import defaultdict

import pandas as pd

from ..utils import (
    clean_quotes,
    get_named_entities,
    handle_input_paths,
    make_answer_mapping
)


@handle_input_paths(
    "raw/bio_instruct_qa_fr/French-full.json",
    "processed/bio_instruct_qa_fr"
)
def bio_instruct_qa_fr(
    input_path: Path,
    output_path: Path,
    return_data: bool = False
) -> List[Path]:
    with input_path.open(encoding="utf-8") as f:
        json_data = json.load(f)
    grouped_by_corpus = defaultdict(list)
    for entry in json_data:
        corpus = entry.get("corpus_name", "UNKNOWN")
        grouped_by_corpus[corpus].append(entry)
    answer_mapping = make_answer_mapping("ABCDE")
    ret = {} if return_data else []
    for corpus_name, entries in grouped_by_corpus.items():
        output_data_per_corpus = []
        for e in entries:
            cleaned_options = {
                key: clean_quotes(value)
                for key, value in e.get("options_translated", {}).items()
            }
            entry = {
                "id": e.get("identifier"),
                "question": clean_quotes(e.get("question_translated")),
                "options": cleaned_options,
                "answer": answer_mapping[e["correct_answer_letter"]],
            }
            output_data_per_corpus.append(entry)
        if return_data:
            ret[corpus_name] = output_data_per_corpus
        else:
            data_filepath = output_path / (corpus_name + ".json")
            with data_filepath.open("w", encoding="utf-8") as f:
                json.dump(output_data_per_corpus, f, indent=2, ensure_ascii=False)
            ret.append(data_filepath)
    return ret


@handle_input_paths(
    "raw/clister/clister_test.tsv",
    "processed/clister"
)
def clister(
    input_path: Path,
    output_path: Path,
    return_data: bool = False
) -> Path:
    df = pd.read_csv(input_path, sep="\t")
    if return_data:
        return df.to_dict(orient="records")
    data_filepath = output_path / input_path.with_suffix(".json").name
    df.to_json(data_filepath, orient="records", indent=2, force_ascii=False)
    return data_filepath


@handle_input_paths(
    "raw/e3c-ner/test.json",
    "processed/e3c-ner"
)
def e3c(
    input_path: Path,
    output_path: Path,
    return_data: bool = False
) -> Path:
    with input_path.open(encoding="utf-8") as f:
        json_data = json.load(f)
    output_data = []
    for key, value in json_data.items():
        transformed_item = {
            "id": key,
            "text": ' '.join(value["text"]),
            "named_entity": get_named_entities(
                value["text"], value["tags"]
            )
        }
        output_data.append(transformed_item)
    if return_data:
        return output_data
    data_filepath = output_path / input_path.name
    with data_filepath.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    return data_filepath
    

@handle_input_paths(
    "raw/frenchmedmcqa/test.json",
    "processed/frenchmedmcqa"
)
def frenchmedmcqa(
    input_path: Path,
    output_path: Path,
    return_data: bool = False
) -> Path:
    with input_path.open(encoding="utf-8") as f:
        json_data = json.load(f)
    output_data = []
    for item in json_data:
        transformed_item = {
            "id": item["id"],
            "question": item["question"],
            "answers": list(item["answers"].values()),
            "correct_answers": item["correct_answers"],
            "subject_name": item["subject_name"],
            "nbr_correct_answers": item["nbr_correct_answers"]
        }
        output_data.append(transformed_item)
    if return_data:
        return output_data
    data_filepath = output_path / input_path.name
    with data_filepath.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    return data_filepath