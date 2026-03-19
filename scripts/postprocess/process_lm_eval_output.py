from pathlib import Path
from argparse import ArgumentParser

import pandas as pd

from partages_llm.eval.post_processing import gather_results_by_domain

DESC="Extract all JSON results from subfolders and save them in a TSV."
INPUT_HELP="Directories (>=1) where the model results are stored"
OUTPUT_HELP="Directory to put the TSV file in"
NAME_HELP="Optional name for the TSV file (without extension)"
BMNO_HELP="Include only model basenames in the output (default is to include 1 parent dir)"


def main():
    parser = ArgumentParser(description=DESC)
    parser.add_argument("input_dir", nargs="+", help=INPUT_HELP)
    parser.add_argument("--output-dir", help=OUTPUT_HELP)
    parser.add_argument("--name-tsv", help=NAME_HELP)
    parser.add_argument("-r", dest="recursive", action="store_true")
    parser.add_argument("-v", dest="verbose", action="store_true")
    parser.add_argument("--bmno", action="store_true", help=BMNO_HELP)
    args = parser.parse_args()
    
    output_dir = args.output_dir if args.output_dir else args.input_dir[0]
    output_filepath_maybe = Path(output_dir) / "gathered.tsv"
    if output_filepath_maybe.exists():
        existing_results = pd.read_csv(output_filepath_maybe, sep='\t')
        output_name = output_filepath_maybe.stem
        write_path = output_filepath_maybe
    else:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        existing_results = None
        output_name = args.name_tsv
        write_path = output_path / (output_name + ".tsv")
    
    results_df = gather_results_by_domain(
        input_dir=args.input_dir,
        output_name=output_name,
        base_model_name_only=args.bmno,
        recursive=args.recursive,
        existing_results=existing_results,
        verbose=args.verbose
    )
    
    results_df.to_csv(write_path, index=False, encoding="utf-8", sep="\t")
    if args.verbose:
        print("\nOutput written to %s" %  write_path)


if __name__ == "__main__":
    main()

