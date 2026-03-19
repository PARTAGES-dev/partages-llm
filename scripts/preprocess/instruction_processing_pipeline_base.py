import os
import json
from uuid import uuid4
from pathlib import Path
from argparse import ArgumentParser

import pandas as pd
from datasets import Dataset

from partages_llm.utils import basic_logger_init, make_version_subdir_path

_DATASET_DIR = Path(os.getenv("HOME")) / "partages-llm-data/mcqa"


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("-v", "--version", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_arguments()
    input_fp = _DATASET_DIR / f"init/v{args.version}"
    logger = basic_logger_init()
    logger.info("Loading data from %s", input_fp)
    df_input = pd.concat([pd.read_parquet(filepath) for filepath in input_fp.glob("*.parquet")])
    with (_DATASET_DIR / "supplementary/specialities-translation.json").open() as f:
        specialities_translation = json.load(f)
    df_processed = df_input.assign(
        doc_id=pd.Series(uuid4().hex for _ in range(len(df_input))).astype(str),
        question=df_input.input.str.replace("\n\t\n", "\n\n"),
        speciality=df_input.instruction.str.extract(
            r". plus particulièrement en ([A-Za-z\s]+).", expand=False
        ).apply(specialities_translation.get).fillna("<N/A>")
    ).drop(["instruction", "input"], axis=1)
    output_dir = _DATASET_DIR / "preproc"
    output_fp = make_version_subdir_path(output_dir, make=True)
    ds = Dataset.from_pandas(df_processed, preserve_index=False)
    logger.info("Built processed dataset: %s", repr(ds))
    ds.save_to_disk(output_fp)
    logger.info("Done; output @ %s", output_fp)


if __name__ == "__main__":
    main()

