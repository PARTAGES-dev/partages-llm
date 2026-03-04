import os
from pathlib import Path
from argparse import ArgumentParser

import pandas as pd

from partages_llm.eval.post_processing import gather_results_by_domain


def main():
    parser = ArgumentParser(
        description="Extract all JSON results from subfolders and save them in a TSV."
    )
    parser.add_argument(
        "input_dir", nargs="+",
        help="Directories where the model results are stored",
    )
    parser.add_argument(
        "--output_dir",
        help="Directory to put the TSV file in",
    )
    parser.add_argument(
        "--name_tsv",
        help="Optional name for the TSV file (without extension)",
    )
    parser.add_argument(
        "--cstats",
        help="Path to a config file for calculating & showing task completion statistics"
    )
    parser.add_argument("-r", dest="recursive", action="store_true")
    parser.add_argument("-v", dest="verbose", action="store_true")
    parser.add_argument("--no-bmno", action="store_true")
    args = parser.parse_args()
    task_group_ref_path =  os.path.normpath(os.path.join(
        os.path.dirname(__file__),
        *[os.pardir] * 3,
        "data/cfg/lm-eval/task-groups-flat.yaml"
    ))
    output_dir = args.output_dir if args.output_dir else args.input_dir[0]
    output_filepath_maybe = Path(output_dir) / "gathered.tsv"
    if output_filepath_maybe.exists():
        existing_results = pd.read_csv(output_filepath_maybe, sep='\t')
        output_name = output_filepath_maybe.stem
    else:
        existing_results = None
        output_name = args.name_tsv
    base_model_name_only = not args.no_bmno
    gather_results_by_domain(
        input_dir=args.input_dir,
        output_dir=output_dir,
        output_name=output_name,
        completion_stats_config=args.cstats,
        task_group_ref_path=task_group_ref_path,
        base_model_name_only=base_model_name_only,
        recursive=args.recursive,
        existing_results=existing_results,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()

