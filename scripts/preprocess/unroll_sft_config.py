import sys
import json
from argparse import ArgumentParser
from typing import Any, Dict
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from partages_llm.training_tools import unroll_config


def dump_unrolled_list(
    data: Dict[str, Any], 
    output_dir_path: Path,
    basename: str
):
    output_path = (output_dir_path / basename).with_suffix(".jsonl")
    with output_path.open("w") as f:
        json.dump(data, f, indent=4)
    return output_path


def main():
    parser = ArgumentParser()
    parser.add_argument("config_filepath")
    parser.add_argument("-c", dest="chunks", type=int, default=1)
    args = parser.parse_args()

    config_filepath = Path(args.config_filepath)
    if not config_filepath.exists():
        raise FileNotFoundError("%s not found" % args.config_filepath)
    
    unrolled_hps_config = unroll_config(config_filepath, return_list=True)
    output_dir_name = config_filepath.stem + "_" + datetime.now().strftime("%y-%m-%d_%H-%M")
    output_dir_path = config_filepath.parents[0] / "unrolled" / output_dir_name
    output_dir_path.mkdir(parents=True, exist_ok=True)
    if args.chunks > 1:
        num_combinations = len(unrolled_hps_config)
        chunk_size = num_combinations // args.chunks
        for idx in tqdm(
            range(0, num_combinations, chunk_size),
            desc=f"Splitting into chunks for {output_dir_path}"
        ):
            idx_upper = min(idx + chunk_size, num_combinations)
            chunk = unrolled_hps_config[idx:idx_upper]
            output_stem = f"params--{idx}-{idx_upper}"
            unrolled_output_path = dump_unrolled_list(chunk, output_dir_path, output_stem)
    else:
        unrolled_output_path = dump_unrolled_list(unrolled_hps_config, output_dir_path, "params")
    sys.exit(unrolled_output_path)


if __name__ == "__main__":
    main()

