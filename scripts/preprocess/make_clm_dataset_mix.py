import os
import json
from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from functools import partial

from datasets import DatasetDict, load_dataset, load_from_disk, concatenate_datasets

from partages_llm.utils import Bunch, basic_logger_init, make_version_subdir_path
from partages_llm.processing import DataMixConfig

_HERE = os.path.dirname(__file__)
_DATADIR_BASE = os.path.join(os.getenv("HOME"), "partages-llm-data")
_TRANSBIO_DATA_FILES = ["train/data-003*-of-00365.arrow"]
_FINEWEB_DATA_FILES = [f"000_0000{i}.parquet"for i in range(6, 10)]
_EXCLUDE_SOURCES = ['WMT16']


def default_filepath_args():
    return Bunch(
        base=os.path.join(_DATADIR_BASE, "clm-corpus/com-clean-dedup/v1"),
        transbio=os.path.join(_DATADIR_BASE, "TransCorpus-bio-fr/v0"),
        parallel=os.path.join(_DATADIR_BASE, "paradocs-subsample/1m"),
        config=os.path.join(_HERE, os.pardir, "configs/clm-corpus-processing/data-mix-config.json"),
        output=os.path.join(_DATADIR_BASE, "clm-corpus/mix")
    )


def subsample_ds(ds, proportion, num_docs_base, seed, logger):
    if isinstance(ds, DatasetDict):
        ds = ds["train"]
    a = int(num_docs_base * proportion)
    if a > ds.num_rows:
        logger.warn(
            "Not enough documents loaded to satisfy proportion %.2f in config: %d < %d",
            proportion, ds.num_rows, a
        )
        return ds
    logger.info("Documents in subsample: %d", a)
    return ds.shuffle(seed=seed).take(a)


def transbio_column_transform_dict(instance):
    return {
        "source": "TransCorpusBio-fr",
        "subset": "none",
        "doc_id": str(hash(instance["text"]))
    }


def fineweb_column_transform_dict(instance):
    return {
        "source": "FineWeb2-HQ",
        "subset": "fra_Latn",
        "doc_id": instance["id"]
    }


def parallel_column_transform_dict(instance):
    return {
        "text": f"English: {instance['src']}\nFrench: {instance['tgt']}",
        "source": "paradocs",
        "subset": "en-fr-strict",
        "doc_id": f"{instance['src_docid']}--{instance['tgt_docid']}"
    }


def normalise_ds(ds, column_transform_dict_func, num_proc):
    def map_func(instance, update_func):
        instance.update(update_func(instance))
        return instance
    remove_columns = [ftr for ftr in ds.features if ftr not in (
        "text", "source", "subset", "doc_id"
    )]
    return ds.map(
        partial(map_func, update_func=column_transform_dict_func),
        num_proc=num_proc,
        remove_columns=remove_columns
    )


def parse_arguments():
    defaults = default_filepath_args()
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    for pathtype in ("base", "transbio", "parallel", "config", "output"):
        parser.add_argument(
            f"--{pathtype}-path",
            type=str,
            default=getattr(defaults, pathtype)
        )
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("-w", "--workers", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_arguments()
    logger = basic_logger_init()
    output_path = make_version_subdir_path(Path(args.output_path), make=True)
    with (output_path / "script_args.json").open("w") as f:
        json.dump(vars(args), f, indent=4)
    logger.info("Loading base dataset from %s", args.base_path)
    base_ds = load_from_disk(args.base_path)
    print(base_ds)
    if _EXCLUDE_SOURCES:
        logger.info("Removing sources: %s", ", ".join(_EXCLUDE_SOURCES))
        num_docs_init = base_ds.num_rows
        base_ds = base_ds.filter(
            lambda instance: instance["source"] not in _EXCLUDE_SOURCES,
            num_proc=args.workers
        )
        logger.info("Done; num_rows %d -> %d", num_docs_init, base_ds.num_rows)
    else:
        logger.info("Base N=%d", base_ds.num_rows)
    ds_list = [base_ds]
    
    with Path(args.config_path).open() as f:
        mix_config_dict = json.load(f)
    mix_config = DataMixConfig(**mix_config_dict)
    logger.info(
        "Data Mix Configuration: \n\t%s",
        "\n\t".join(f"{k}: {v}" for k, v in mix_config_dict.items())
    )
    with (output_path / "mix-config.json").open("w") as f:
        json.dump(mix_config_dict, f, indent=4)
     
    makesample = partial(subsample_ds, num_docs_base=base_ds.num_rows, seed=args.seed, logger=logger)
    normalise_ds_w = partial(normalise_ds, num_proc=args.workers)
    
    logger.info("Loading TransBio dataset from %s", args.transbio_path)
    if not mix_config.transbio_proportion:
        logger.info("Sample proportion=%s; skipping", mix_config.transbio_proportion)
    else:
        transbio_subset = makesample(
            load_dataset(args.transbio_path, data_files=_TRANSBIO_DATA_FILES),
            proportion=mix_config.transbio_proportion
        )
        logger.info("Running column normalisation...")
        ds_list.append(normalise_ds_w(
            ds=transbio_subset, column_transform_dict_func=transbio_column_transform_dict
        ))

    logger.info("Loading FineWeb dataset")
    if not mix_config.fineweb_proportion:
        logger.info("Sample proportion=%s; skipping", mix_config.fineweb_proportion)
    else:
        fineweb_subset = makesample(
            load_dataset("epfml/FineWeb2-HQ", data_dir="fra_Latn", data_files=_FINEWEB_DATA_FILES),
            proportion=mix_config.fineweb_proportion
        )
        logger.info("Running column normalisation...")
        ds_list.append(normalise_ds_w(
            ds=fineweb_subset, column_transform_dict_func=fineweb_column_transform_dict
        ))

    logger.info("Loading ParaDocs dataset from %s", args.parallel_path)
    if not mix_config.paradocs_proportion:
        logger.info("Sample proportion=%s; skipping", mix_config.paradocs_proportion)
    else:
        parallel_subset = makesample(
            load_from_disk(args.parallel_path),
            proportion=mix_config.paradocs_proportion
        )
        logger.info("Running column normalisation...")
        ds_list.append(normalise_ds_w(
            ds=parallel_subset, column_transform_dict_func=parallel_column_transform_dict
        ))

    ds_mix = concatenate_datasets(ds_list).shuffle(seed=args.seed)
    logger.info("Mixed corpus finished: %s\nSaving to disk @ %s", repr(ds_mix), output_path)
    ds_mix.save_to_disk(output_path)
    print("=" * 50)


if __name__ == "__main__":
    main()

