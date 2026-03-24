import os
import json
from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from datetime import datetime
from functools import partial
from tqdm import tqdm
from typing import Any, Dict

from datasets import load_from_disk

from partages_llm.utils import basic_logger_init, make_version_subdir_path
from partages_llm.processing import clean_text

_DATADIR_BASE = Path(os.getenv("HOME")) / "partages-llm-data"
URV_HELP = "Include this flag to use the version of the corpus that includes documents not "\
"licensed for downstream commercial use (the 'research only' version)"
WORKERS_HELP = "The number of parallel processes to use in applyint the text cleaning function"
WCM_HELP = "The number of valid words a document has to have to be included in the output"


def parse_arguments():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-v", "--dataset-version", type=int, default=0)
    parser.add_argument("-r", "--use-research-version", action="store_true", help=URV_HELP)
    parser.add_argument("-w", "--workers", type=int, default=16, help=WORKERS_HELP)
    parser.add_argument("-m", "--word-check-min", type=int, default=3, help=WCM_HELP)
    return parser.parse_args()


def text_cleaner_map(instance: Dict[str, Any], word_check_min: int):
    return clean_text(instance["text"], strict=True, word_check_min=word_check_min)


def main():
    args = parse_arguments()
    logger = basic_logger_init()
    dataset_type = "research" if args.use_research_version else "com"
    data_dir = Path(_DATADIR_BASE) / f"wp2-corpus/{dataset_type}/v{args.dataset_version}"
    data_path = data_dir / "train"
    if not data_path.exists():
        data_path = data_dir
    logger.info("Loading from %s", data_path)
    ds = load_from_disk(data_path)
    logger.info("Applying `clean_text`...")
    func = partial(text_cleaner_map, word_check_min=args.word_check_min)
    ds_clean = ds.map(func, num_proc=args.workers)
    n_removed = 0
    for instance in tqdm(ds_clean, desc="Checking outputs"):
        if instance["text"] and not instance["text_cleaned"]:
            n_removed += 1
    pc_removed = 100 * (n_removed / ds.num_rows)
    logger.info("%d of %d documents removed altogether (%.2f%%)", n_removed, ds.num_rows, pc_removed)
    ds_clean = ds_clean.remove_columns("text").rename_column("text_cleaned", "text")
    output_dir = data_dir.parents[1] / (data_dir.parents[0].name + "-clean")
    output_path = make_version_subdir_path(output_dir)
    ds_clean.save_to_disk(output_path)
    arg_dict = vars(args)
    arg_dict["script"] = __file__
    arg_dict["run_finished_at"] = datetime.now().ctime()
    arg_dict["input_path"] = str(data_path)
    with (output_path / "script_params.json").open("w") as f:
        json.dump(arg_dict, f, indent=4)
    logger.info("Done; output @ %s\n%s", output_path, "=" * 75)


if __name__ == "__main__":
    main()

