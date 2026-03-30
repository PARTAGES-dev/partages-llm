import os
import json
import traceback
from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
from typing import Callable, Optional, Union
from logging import RootLogger
from functools import partial
from datetime import datetime

from datasets import Dataset, DatasetDict, Features, load_from_disk
from transformers import AutoTokenizer

from partages_llm.utils import basic_logger_init, ndnt
from partages_llm.processing import (
    ValidationSplitConfig,
    get_tokenized_ds_features,
    generate_concatenated_tokenized_ds,
    filter_tokenized_ds
)

_DATADIR_BASE = Path(os.getenv("HOME")) / "partages-llm-data"
_DS_CACHE =  _DATADIR_BASE / "hf-cache"
_DATASET_TYPE2DIR = {
    "original": "com",
    "clean": "com-clean",
    "dedup": "com-clean-dedup",
    "mix": "mix"
}
_EXCLUDE_SOURCES = [
    'WMT16'
]

DESC = "Step 1 of the preprocessing pipeline for the PARCOMED CLM corpus."
DATASETTYPE_HELP = "The variant of the PARCOMED dataset to be processed. "\
"This will determine the subirectory of the data directory where the script "\
"will look for the input dataset. Options: "\
"original: unprocessed corpus "\
"| clean: output of `clean_clm_dataset` "\
"| dedup: output of `clean_clm_dataset` followed by `deduplicate_clm_dataset` "\
"| mix: output of `make_clm_dataset_mix`"
MML_HELP = ""
MINLENGTH_HELP = "Lower bound on the number of tokens in a processed sequence "\
"to consider. Note that this includes special tokens, but not padding."
WORKERS_HELP = "Number of parallel processes to use in the mapping and filtering "\
"functions."
ESC_HELP = "Path to a JSON file specifying the parameters to use for the train/"\
"validation split."
OVERFLOW_HELP = "When truncating sequences, the last n tokens, n determined by "\
"`stride`, will be included in the subsequent sequence."
CONCATENATE_HELP = "Include this flag to concatenate adjacent documents into "\
"contiguous sequences - this ensures that all sequences (except potentially the "\
"last one) will have the same length."
URV_HELP = "Include this flag to use the version of the corpus that includes documents not "\
"licensed for downstream commercial use (the 'research only' version)"
CTMN_HELP = "Use a shortened version of the tokenizer name for the output dataset."


def parse_arguments() -> Namespace:
    default_output_dir = str(_DATADIR_BASE / "tokens")
    default_eval_set_config_path = os.path.normpath(os.path.join(
        os.path.dirname(__file__), os.pardir,
        "configs/clm-corpus-processing/validation-set-config.json"
    ))
    parser = ArgumentParser(description=DESC, formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("tokenizer_name_or_path")
    parser.add_argument("-v", "--dataset-version", type=int, default=0)
    parser.add_argument("-t", "--dataset-type", choices=tuple(_DATASET_TYPE2DIR), default="original", help=DATASETTYPE_HELP)
    parser.add_argument("-l", "--model-max-length", type=int, default=2048, help=MML_HELP)
    parser.add_argument("-m", "--min-length", type=int, default=5, help=MINLENGTH_HELP)
    parser.add_argument("-o", "--output-dir", default=default_output_dir)
    parser.add_argument("-w", "--workers", type=int, default=16, help=WORKERS_HELP)
    parser.add_argument("-p", "--pad-token", default="<pad>")
    parser.add_argument("-b", "--bos-token")
    parser.add_argument("-d", "--eval-set-config-path", type=Path, default=default_eval_set_config_path, help=ESC_HELP)
    parser.add_argument("-f", "--overflow", action="store_true", help=OVERFLOW_HELP)
    parser.add_argument("-C", "--concatenate", action="store_true", help=CONCATENATE_HELP)
    parser.add_argument("-s", "--stride", type=int, default=4)
    parser.add_argument("-r", "--use-research-version", action="store_true", help=URV_HELP)
    parser.add_argument("-c", "--cut-tokenizer-model-name", action="store_true", help=CTMN_HELP)
    return parser.parse_args()


def make_val_split(ds, config, logger=None) -> Dataset:
    disp = logger.info if logger else print
    if config is None:
        return ds
    disp("Creating validation split")
    test_size = config.num_validation_docs if config.num_validation_docs else config.proportion
    class_encoded_ds = ds.class_encode_column("source").train_test_split(
        test_size=test_size,
        stratify_by_column="source",
        seed=config.seed
    )
    return class_encoded_ds


def run_tokenization(
    dataset_or_path: Union[str, Path, Dataset, DatasetDict],
    tokenize_func: Callable,
    overflow: int,
    num_proc: int,
    features: Features,
    min_length: int,
    concatenate_generator_func: Callable,
    logger: Optional[RootLogger] = None
):
    """
    Core tokenization logic - handles the tokenizer call, length-based filtering
    (via `filter_tokenized_ds`) and concatenation.
    """
    disp = logger.info if logger else print
    ds = dataset_or_path if isinstance(dataset_or_path, (Dataset, DatasetDict)) \
        else load_from_disk(dataset_or_path)        
    disp("Running tokenization")
    remove_columns = ds.column_names
    if isinstance(ds, DatasetDict):
        remove_columns = remove_columns["train"]
    tokenized_ds = ds.map(
        partial(tokenize_func, overflow=overflow),
        num_proc=num_proc,
        remove_columns=remove_columns,
        batched=True,
        features=features,
        desc="=>tokenizer=>"
    )
    disp("Removing sequences shorter than %d" % min_length)
    if isinstance(tokenized_ds, Dataset):
        tokenized_ds_filtered = filter_tokenized_ds(tokenized_ds, min_length)
        disp(
            "Done filtering: num_rows %d -> %d" % (
                tokenized_ds.num_rows,
                tokenized_ds_filtered.num_rows
            )
        )
    else:
        tokenized_ds_filtered = DatasetDict({
            "train": filter_tokenized_ds(tokenized_ds["train"], min_length),
            "val": filter_tokenized_ds(tokenized_ds["test"], min_length)
        })
        disp(
            "Done filtering: num_rows train %d -> %d; num_rows val. %d -> %d" % (
                tokenized_ds.num_rows["train"],
                tokenized_ds_filtered.num_rows["train"],
                tokenized_ds.num_rows["test"],
                tokenized_ds_filtered.num_rows["val"],
            )
        )
    if concatenate_generator_func:
        disp("Building concatenated version...")
        if isinstance(tokenized_ds_filtered, Dataset):
            tokenized_ds_output = Dataset.from_generator(
                partial(concatenate_generator_func, ds=tokenized_ds_filtered),
                cache_dir=_DS_CACHE
            )
            disp(
                "Finished concatenation: num_rows %d -> %d" % (
                    tokenized_ds_filtered.num_rows,
                    tokenized_ds_output.num_rows
                )
            )
        else:
            tokenized_ds_output = DatasetDict({
                "train": Dataset.from_generator(
                    partial(
                        concatenate_generator_func, ds=tokenized_ds_filtered["train"]
                    ), cache_dir=_DS_CACHE
                ),
                "val": Dataset.from_generator(
                    partial(
                        concatenate_generator_func, ds=tokenized_ds_filtered["val"]
                    ), cache_dir=_DS_CACHE
                )
            })
            disp(
                "Finished concatenation: num_rows train %d -> %d; num_rows val. %d -> %d" % (
                    tokenized_ds_filtered.num_rows["train"],
                    tokenized_ds_output.num_rows["train"],
                    tokenized_ds_filtered.num_rows["val"],
                    tokenized_ds_output.num_rows["val"],
                )
            )
    else:
        tokenized_ds_output = tokenized_ds_filtered
    return tokenized_ds_output
    

def build_tokenized_dataset(args: Namespace, logger: RootLogger):
    ### general args/metadata setup
    arg_dict = vars(args)
    if args.eval_set_config_path:
        with open(args.eval_set_config_path) as f:
            validation_split_config_dict = json.load(f)
        arg_dict["validation_split_config"] = validation_split_config_dict
        validation_split_config = ValidationSplitConfig(**validation_split_config_dict)
    else:
        validation_split_config = None
    arg_str = ndnt(2).join(f"{k}: {v}" for k, v in arg_dict.items())
    output_dir = Path(args.output_dir)
    
    ### tokenizer functionality
    logger.info("TOKENIZATION RUN\n\tParameters:%s%s", ndnt(2), arg_str)
    assert output_dir.exists(), f"{output_dir} isn't a valid directory path"
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path, model_max_length=args.model_max_length,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": args.pad_token})
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.bos_token_id is None and args.bos_token is not None:
        tokenizer.add_special_tokens({"bos_token": args.bos_token})
    logger.info("Loaded tokenizer: %s", type(tokenizer).__name__)
    def _tknz_func(instance, overflow):
        batch_encoding = tokenizer(
            instance["text"],
            padding=False,
            add_special_tokens=True,
            truncation=True,
            max_length=args.model_max_length,
            return_tensors="np",
            return_special_tokens_mask=True,
            return_overflowing_tokens=overflow,
            stride=args.stride
        )
        if overflow:
            del batch_encoding["overflow_to_sample_mapping"]
        return batch_encoding
    
    ### input
    dataset_type = _DATASET_TYPE2DIR[args.dataset_type]
    if args.use_research_version:
        dataset_type = dataset_type.replace("com", "research")
    data_dir = _DATADIR_BASE / f"wp2-corpus/{dataset_type}/v{args.dataset_version}"
    logger.info("Loading from %s", data_dir)
    ds = load_from_disk(data_dir)
    if _EXCLUDE_SOURCES:
        logger.info("Removing sources: %s", ", ".join(_EXCLUDE_SOURCES))
        num_docs_init = ds.num_rows
        ds = ds.filter(
            lambda instance: instance["source"] not in _EXCLUDE_SOURCES,
            num_proc=args.workers
        )
        logger.info("Done; num_rows %d -> %d", num_docs_init, ds.num_rows)
    tokenized_ds_features = get_tokenized_ds_features()
    ds = make_val_split(ds, validation_split_config, logger)
    concatenate_generator_func = partial(
        generate_concatenated_tokenized_ds,
        sequence_length=args.model_max_length,
        bos_token_id=tokenizer.bos_token_id,
        space_id=tokenizer.encode(" ")[-1],
        stride=args.stride,
        minimum_remainder=args.min_length,
        return_remainder=False,
        verbose=True
    ) if args.concatenate else None
    
    ### output naming setup
    tokenized_ds_name_base = Path(args.tokenizer_name_or_path).name
    if args.cut_tokenizer_model_name:
        tokenized_ds_name_base = tokenized_ds_name_base.split("-")[0]
    tokenized_ds_name = tokenized_ds_name_base + \
        f"_wp2-{dataset_type}-v{args.dataset_version}-"
    if args.concatenate:
        tokenized_ds_name += "cc-"
    tokenized_ds_name += str(args.model_max_length)
    if args.overflow:
        tokenized_ds_name += f"-ovf{args.stride}"
    tokenized_ds_dir = output_dir / tokenized_ds_name
    
    ### let's go
    processed_ds = run_tokenization(
        ds,
        _tknz_func,
        args.overflow,
        args.workers,
        tokenized_ds_features,
        args.min_length,
        concatenate_generator_func,
        logger
    )
    tokenized_ds_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Writing tokenized dataset to disk @ %s" % tokenized_ds_dir)
    processed_ds.save_to_disk(tokenized_ds_dir)

    ### wrapping up
    arg_dict["script"] = __file__
    arg_dict["run_finished_at"] = datetime.now().ctime()
    arg_dict["eval_set_config_path"] = str(arg_dict["eval_set_config_path"])  # pathlib.Path can't be serialized to JSON
    with (tokenized_ds_dir / "script_params.json").open("w") as f:
        json.dump(arg_dict, f, indent=4)
    logger.info("Finished\n%s", "=" * 50)


def main():
    logger = basic_logger_init()
    args = parse_arguments()
    try:
        build_tokenized_dataset(args, logger)
    except Exception:
        traceback.print_exc()
        logger.error("Script failed due to above exception")


if __name__ == "__main__":
    main()

