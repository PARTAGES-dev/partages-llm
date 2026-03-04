import os
from pathlib import Path
from argparse import ArgumentParser

from datasets import Features, load_dataset

from partages_llm.utils import basic_logger_init, make_version_subdir_path

_DATADIR_BASE = Path(os.getenv("HOME")) / "partages-llm-data"
_DS_CACHE =  _DATADIR_BASE / "hf-cache"


def main():
    parser = ArgumentParser()
    parser.add_argument('-r', dest='get_research_version', action='store_true')
    parser.add_argument('-i', dest='instruction_tuning', action='store_true')
    parser.add_argument('--token')
    args = parser.parse_args()
    logger = basic_logger_init()

    load_kwargs = {'cache_dir': _DS_CACHE}
    if args.token is None:
        logger.warning("No Hugging Face access token provided, will try to download dataset without one")
    else:
        load_kwargs["token"] = args.token
    if args.instruction_tuning:
        load_kwargs['data_dir'] = 'instruction-tuning'
        column_names = [
            'instruction',
            'input',
            'output',
            'source',
            'data_dir',
            'source_split'
        ]
        load_kwargs["features"] = Features.from_dict({
            column_name: {
                'dtype': 'string',
                '_type': 'Value'
            } for column_name in column_names
        })
    else:
        load_kwargs['data_dir'] = "fine-tuning"
    ds_id = "LIMICS/PARTAGES"
    if args.get_research_version:
        ds_id += '-Research'
    ds = load_dataset(ds_id, **load_kwargs)
    logger.info('Dataset downloaded:\n%s', repr(ds))
    if args.instruction_tuning:
        write_dir = _DATADIR_BASE / 'wp2-instructions' / 'init'
    else:
        ds_out = ds['train'].remove_columns(
            ['instruction', 'output']
        ).rename_column('input', 'text')
        write_dir = _DATADIR_BASE / "wp2-corpus"/ (
            'research' if args.get_research_version else 'com'
        )
    output_dir = make_version_subdir_path(write_dir, make=True)
    ds_out.save_to_disk(output_dir, max_shard_size='250MB')
    logger.info('OUTPUT\n%s\n-> Saved to %s\n%s', 
        repr(ds_out), output_dir, '=' * 50
    )


if __name__ == "__main__":
    main()

