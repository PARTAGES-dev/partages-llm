import os
import json
import tempfile
import subprocess
import multiprocessing as mp
from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
from typing import Any, Dict, List, Optional, Tuple, Union
from _io import TextIOWrapper
from logging import RootLogger
from uuid import uuid4
from functools import partial
from itertools import product, repeat

from datasets import Dataset, DatasetDict, load_from_disk
from tqdm import tqdm
from lxml import etree

from partages_llm.utils import basic_logger_init, make_version_subdir_path

_DATADIR_BASE = Path(os.getenv("HOME")) / "partages-llm-data"
DESC = "Optional step 2 of the preprocessing pipeline for the PARCOMED CLM corpus: "\
"deduplicates the text using the Onion corpus processing tool."
NGRAM_HELP = "Length of n-grams to compare"
THRESHOLD_HELP = "Similarity threshold to define duplication"
BUFFER_HELP = "Set buffer size in bytes"
DOCLIMIT_HELP = "Ceiling on the number of documents to include"
MSPVD_HELP = "Ceiling on the number of sentences to include in each .vert file to "\
"input to onion"
Q_HELP = "Suppress terminal output"
SIVF_HELP = "Save the file(s) passed to onion; default is to keep in a temporary "\
"directory that is deleted when the run wraps up"
SHUFFLECORPUS_HELP = "Randomly shuffles the dataset before launching processing"
STATMODE_HELP = "Runs the script in `statistics mode` - doesn't write any data to "\
"disk, just collects measurements on how much the input is compressed for a range "\
"of parameters (multiple )"
CDC_HELP = "Remove the buildup of datasets library cache files from the output directory"
WORKERS_HELP = "Ceiling on the number of parallel processes to run in statistics mode"


def parse_arguments() -> Namespace:
    default_output_dir = str(_DATADIR_BASE / "parcomed")
    parser = ArgumentParser(description=DESC, formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("corpus_path", type=str)
    parser.add_argument("-n", dest="ngram", type=int, nargs="*", default=3, help=NGRAM_HELP)
    parser.add_argument("-t", dest="threshold", type=float, nargs="*", default=.5, help=THRESHOLD_HELP)
    parser.add_argument("-b", dest="buffer", type=int, default=16777216 * 4, help=BUFFER_HELP)
    parser.add_argument("-l", dest="doc_limit", type=int, default=int(1e9), help=DOCLIMIT_HELP)
    parser.add_argument("-m", dest="max_sentences_per_vert_doc", type=int, default=100, help=MSPVD_HELP)
    parser.add_argument("-q", dest="silent", action="store_true", help=Q_HELP)
    parser.add_argument("-v", dest="save_input_vert_file", action="store_true", help=SIVF_HELP)
    parser.add_argument("-s", dest="shuffle_corpus", action="store_true", help=SHUFFLECORPUS_HELP)
    parser.add_argument("-S", dest="stat_mode", action="store_true", help=STATMODE_HELP)
    parser.add_argument("-c", dest="clear_ds_cache", action="store_true", help=CDC_HELP)
    parser.add_argument("-w", dest="max_workers", type=int, default=1, help=WORKERS_HELP)
    parser.add_argument("-o", dest="output_dir", type=str, default=default_output_dir)
    return parser.parse_args()


def vert_doc_template(doc_id: int, name: str, content: str) -> str:
    """
    Atomic unit of .vert file content generation.

    Args:
        doc_id: Integer ID for the document - usually just a counter.
        name: String ID.
        content: Contents of the docuement.

    Returns:
        The text data of `content` formatted for deduplication.
    """
    doc = f'\n<doc id="{doc_id}" name="{name}">'
    if isinstance(content, list):
        for p in content:
            doc += f'\n<p>\n{"\n".join(p.split())}\n</p>'
    elif isinstance(content, str):
        doc += f'\n<p>\n{"\n".join(content.split())}\n</p>'
    doc += "\n</doc>"
    return doc


def count_lines(filepath: Union[str, Path]) -> int:
    """
    Simply wraps the bash `wc -l` command in a subprocess and returns the number of lines in the given file 
    
    Args:
        filepath: File with lines to be counted.

    Returns:
        The number of lines in the file.
    """
    if not isinstance(filepath, str):
        filepath = str(filepath)
    return int(subprocess.check_output(["wc", "-l", filepath]).split()[0])


def template_wo(f_io: TextIOWrapper, doc: Dict[str, Any], idx: int, content: str) -> int:
    """
    Wraps `vert_doc_template` to write the output to disk.

    Args:
        f_io: Output .vert file opened in write mode.
        doc: Dataset instance to be processed.
        idx: Counter for the current function call.
        content: Text data to be formatted.

    Returns:
        Incremented counter: `idx + 1`.
    """
    source = doc.get("source", "partages-wp2")
    subset = doc.get("subset", "train")
    name_elems = f"doc{idx}", source, subset, uuid4().hex
    name = "__".join(name_elems)
    f_io.write(vert_doc_template(idx, name, content))
    return idx + 1


def build_and_write_vert_file(
    input_dir_path: Path,
    ds: Dataset,
    doc_limit: int,
    disable_pb: bool, 
    max_sentences_per_vert_doc: int,
) -> Tuple[Path, int]:
    """
    Preprocessing function for deduplication: builds a file that Onion will accept as input.

    Args:
        input_dir_path: Directory in which to put the generated file.
        ds: Dataset to be converted to .vert.
        doc_limit: Ceiling on the number of dataset instances to be considered.
        disable_pb: Suppress progress bar output
        max_sentences_per_vert_doc: Write to the output file in chunks of this size.

    Returns:
        A 2-tuple: the filepath into which the output was written, and the number of documents therein
    """
    vert_file_path_input = input_dir_path / "corpus.vert"
    tqdm_total = min(ds.num_rows, doc_limit)
    tqdm_desc = f"Building input file {vert_file_path_input.name}"
    with vert_file_path_input.open("w") as f:
        vert_docs_count = 0
        for doc in tqdm(ds, disable=disable_pb, total=tqdm_total, desc=tqdm_desc):
            if not doc["text"]:
                continue
            if vert_docs_count >= doc_limit:
                break
            if doc["text"].count(". ") >= 2:
                text = [sentence + "." for sentence in doc["text"].split(". ")]
                n_sent = len(text)
                if n_sent > max_sentences_per_vert_doc:
                    for j in range(0, n_sent, max_sentences_per_vert_doc):
                        k = min(j + max_sentences_per_vert_doc, n_sent)
                        vert_docs_count = template_wo(f, doc, vert_docs_count, text[j:k])
                else:
                    vert_docs_count = template_wo(f, doc, vert_docs_count, text)
            else:
                vert_docs_count = template_wo(f, doc, vert_docs_count, doc["text"])
    return vert_file_path_input, vert_docs_count


def get_counts_vert(f_io: TextIOWrapper) -> Dict[str, int]:
    """
    Calculates some basic info about the given .vert file.

    Args:
        f_io: TextIOWrapper object opened in read mode.

    Returns:
        Dictionary with counts of the number of documents, bytes, and tokens in the file. 
    """
    n_docs = n_tokens = n_bytes = 0
    for line in f_io:
        if line.startswith("<doc id="):
            n_docs += 1
        elif not line.startswith("<"):
            n_tokens += 1
            n_bytes += len(line.encode("utf-8"))
    return {"docs": n_docs, "tokens": n_tokens, "bytes": n_bytes}


def run_onion(
    input_path: Union[Path, str],
    executable: str,
    opts: str,
    output_dir_path: Path,
    logger: Optional[RootLogger] = None,
    log_prefix: str = ""
) -> Path:
    """
    Calls the Onion deduplication tool on the filepath provided.

    Args:
        input_path: Path to a formatted .vert file to be deduplicated.
        executable: Path to the Onion executable.
        opts: Command-line arguments to pass to Onion.
        output_dir_path: Where to put the deduplicated file - will put a file called `corpus-dedup.vert`
                        in this folder; note that an existing file with the same path will be overwritten.
        logger: Optionally pass a RootLogger object - if not provided, will print messages to stdout.
        log_prefix: Optional prefix string for logging messages

    Returns:
        The path to the output file - `[output_dir_path]/corpus-dedup.vert`
    """
    if isinstance(input_path, Path):
        input_path = str(input_path)
    disp = logger.info if logger else print
    output_path = output_dir_path / "corpus-dedup.vert"
    dedup_cmd = [executable, *opts.split(), input_path]
    dedup_cmd_str = " ".join(dedup_cmd)
    disp_args_cmd = log_prefix, dedup_cmd_str, output_path
    disp("%sRunning deduplication command: %s > %s" % disp_args_cmd)
    output_path_f = output_path.open("w")
    returncode = subprocess.call(dedup_cmd, stdout=output_path_f)
    disp_args_return = log_prefix, returncode
    disp("%sReturn code %d" % disp_args_return)
    return output_path


def statmode_func(
    ngram: int,
    threshold: int,
    buffer: int,
    run_kwargs: Dict[str, Any]
) -> Dict[str, int]:
    """
    Calls Onion without writing to disk: to be used for collecting statistics.

    Args:
        ngram: Value to pass to the `-n` command line argument.
        threshold: Value to pass to the `-t` command line argument.
        buffer: Value to pass to the `-b` command line argument.
        run_kwargs: Keyword arguments for `run_onion`.

    Returns:
        Dictionary with statistics about the deduplication run, à la `get_counts_vert`.
    """
    opts = f"-qsmn {ngram} -t {threshold} -b {buffer}"
    output_filepath = run_onion(opts=opts, **run_kwargs)
    dedup_output_counts = get_counts_vert(output_filepath.open())
    dedup_output_counts.update({"ngram": ngram, "threshold": threshold})
    return dedup_output_counts


def statmode_func_mp_wrapper(
    process_id_offset: int,
    ngram: int,
    threshold: float,
    buffer: int,
    tmp_path: Path,
    run_kwargs: Dict[str, Any]
) -> Dict[str, int]:
    """
    Wrapper for `statmode_func` that manages the `run_onion` kwargs for a multiprocessing context.
    
    Args:
        process_id_offset: Integer by which to offset the current process ID - if there are more 
                            processes to run than cores specified, the the multiprocessing pools
                            are chunked and the run IDs need to be offset by the chunk size.
        ngram: See Onion arguments.
        threshold: See Onion arguments.
        buffer: See Onion arguments.
        tmp_path: Where to make the process-specific subdirectory.
        run_kwargs: Keyword arguments for `run_onion`.

    Returns:
        See `statmode_func`.
    """
    proc = mp.current_process()._identity[0] + process_id_offset
    output_dir_path = tmp_path / f"output-{proc}"
    output_dir_path.mkdir()
    run_kwargs.update({
        "log_prefix": f"Process {proc}: ",
        "output_dir_path": output_dir_path
    })
    return statmode_func(ngram, threshold, buffer, run_kwargs)


def vert2xml(
    f_io: TextIOWrapper,
    root_name: str = "corpus",
    disable_pb: bool = False,
    total_lines: Optional[int] = None
) -> List[str]:
    """
    Converts a .vert file to XML.

    Args:
        f_io: .vert file opened in read mode.
        root_name: What to call the root node of the XML tree structure.
        disable_pb: Suppress progress bar output
        total_lines: The number of lines to be read: only used by `tqdm` if activated.

    Returns:
        List of `f_io` contents formatted for XML.
    """
    xml_esc = str.maketrans({
        "\n": " ",
        "<": r"&lt;",
        ">": r"&gt;",
        "&": r"&amp;",
        '"': r"&quot;", 
        "'": r"&apos;"
    })
    xml_lines = [
        '<?xml version="1.0" encoding="utf-8"?>\n',
        f"<{root_name}>\n"
    ]
    for line in tqdm(f_io, disable=disable_pb, total=total_lines, desc="Formatting output"):
        if line.startswith(("<doc id=", "</doc>", "</p>")):
            xml_lines.append(line)
        elif line.startswith("<p>"):
            xml_lines.append(line.replace("\n", ""))
        elif line == "\n":
            continue
        else:  # actual text data (tokens)
            xml_lines.append(line.translate(xml_esc))
    xml_lines.append(f"</{root_name}>")
    return xml_lines


def tree_parse_generator(xml_path: Union[str, Path]) -> Dict[str, str]:
    """
    Generator function to be passed to `Dataset.from_generator` to build a text dataset from an XML tree.

    Args:
        xml_path: Path to the XML file to read.

    Yields:
        Dictionary containing the `source`, `subset` and `doc_id` metadata from PARCOMED, along with the
        relevant (deduplicated, usually) text data, to be converted to a Dataset element.
    """
    for document in etree.parse(xml_path).findall("doc"):
        metadata = document.attrib["name"].split("__")
        dict_ = {
            "source": metadata[1],
            "subset": metadata[2],
            "doc_id": metadata[3],
            "text": " ".join(p.text for p in document.findall("p") if p.text)
        }
        yield dict_


def main():

    ## SETUP ## 
    args = parse_arguments()
    log_level = "error" if args.silent else "info"
    logger = basic_logger_init(log_level)
    if os.path.isdir(args.corpus_path):
        logger.info("Loading data from %s", args.corpus_path)
        ds = load_from_disk(args.corpus_path)
        if isinstance(ds, DatasetDict):
            ds = ds["train"]
        if args.shuffle_corpus:
            ds = ds.shuffle(seed=976431)
        vert_file_path_input = None
        logger.info("Dataset loaded:\n%s", repr(ds))
    elif os.path.isfile(args.corpus_path) and args.corpus_path.endswith(".vert"):
        logger.info("Input dataset already processed; starting from checkpoint @ %s", args.corpus_path)
        vert_file_path_input = Path(args.corpus_path)
    tempfile.tempdir = (Path(__file__).parents[0] / "../../../data/tmp").resolve()
    tempfile.tempdir.mkdir(exist_ok=True)
    onion_exec = os.getenv("HOME") + "/onion"
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir.decode())

        if vert_file_path_input is None:
            ## CONSTRUCT DEDUPLICATION INPUT ##
            vert_file_loc_input = Path(args.corpus_path) if args.save_input_vert_file else tmp_path
            vert_file_path_input, vert_docs_count = build_and_write_vert_file(
                input_dir_path=vert_file_loc_input,
                ds=ds,
                doc_limit=args.doc_limit,
                disable_pb=args.silent,
                max_sentences_per_vert_doc=args.max_sentences_per_vert_doc
            )
        else:
            vert_docs_count = count_lines(vert_file_path_input)
        input_st_size = vert_file_path_input.stat().st_size
        logger.info("Input file size: documents %d, bytes %d", vert_docs_count, input_st_size)
        
        ## DEDUPLICATE ##
        if args.stat_mode:
            # calculate descriptive tasks about duplicates in the corpus to help choose parameters:
            # doesn't actually build a deduplicated corpus
            output = {
                "original": get_counts_vert(vert_file_path_input.open()),
                "dedup_runs": []
            }
            run_onion_kwargs = {
                "input_path": vert_file_path_input,
                "executable": onion_exec,
                "logger": logger
            }
            total_runs = len(args.ngram) * len(args.threshold)
            num_proc = min(args.max_workers, total_runs)
            if num_proc > 1:
                params = list(product(args.ngram, args.threshold))
                map_func = partial(
                    statmode_func_mp_wrapper,
                    buffer=args.buffer,
                    tmp_path=tmp_path,
                    run_kwargs=run_onion_kwargs
                )
                with mp.Pool(num_proc) as pool:
                    for i, lower_arg_idx in enumerate(range(0, total_runs, num_proc)):
                        upper_arg_idx = min(lower_arg_idx + num_proc, total_runs)
                        params_this_iter = params[lower_arg_idx:upper_arg_idx]
                        process_id_offset = i * num_proc
                        map_args = [(j, *t) for j, t in zip(repeat(process_id_offset), params_this_iter)]
                        output["dedup_runs"].extend(pool.starmap(map_func, map_args))
            else:
                run_onion_kwargs["output_dir_path"] = tmp_path
                for ngram, threshold in product(args.ngram, args.threshold):
                    output["dedup_runs"].append(
                        statmode_func(ngram, threshold, args.buffer, run_onion_kwargs)
                    )
        else:
            onion_opts = f"-smn {args.ngram[0]} -t {args.threshold[0]} -b {args.buffer}"
            if args.silent:
                onion_opts += " -q"
            vert_file_path_output = run_onion(vert_file_path_input, onion_exec, onion_opts, tmp_path, logger)
            output_st_size = vert_file_path_output.stat().st_size
            percent_shrinkage = 100 - (100 * output_st_size / input_st_size)
            logger.info(
                "Output file %s size in bytes: %d - shrunk by %.1f%%",
                vert_file_path_output.name, output_st_size, percent_shrinkage
            )
            total_lines = count_lines(vert_file_path_output)
            
            ## FORMAT OUTPUT ##
            with vert_file_path_output.open("r") as f:
                xml_lines = vert2xml(f, disable_pb=args.silent, total_lines=total_lines)
            xml_path = vert_file_path_output.parents[0] / vert_file_path_output.name.replace("vert", "xml")
            logger.info("Writing to %s...", xml_path)
            with xml_path.open("w") as f:
                f.write("".join(xml_lines))
            logger.info("Generating Dataset from XML parser...")
            try:
                output = Dataset.from_generator(partial(tree_parse_generator, xml_path=xml_path))
            except Exception as exc:
                logger.error("%s encountered: dumping vert & XML files...", type(exc).__name__)
                for p in (vert_file_path_output, xml_path):
                    os.rename(p, tempfile.tempdir / "dedup-debug-dump" / p.name)
                raise exc
        logger.info("Discarding temporary directory...")
    
    ## SAVE RESULTS ##
    logger.info("\n** OUTPUT **\n%s", repr(output))
    for input_parent in Path(args.corpus_path).parents:
        if not input_parent.name.startswith(("train", "v")):
            output_dataset_type = input_parent.name
            break
    output_path_name = output_dataset_type + "-dedup"
    output_path = Path(args.output_dir) / output_path_name
    output_path.mkdir(exist_ok=True)
    stem = "xp" if args.stat_mode else "v"
    output_path = make_version_subdir_path(output_path, make=True, stem=stem)
    if isinstance(output, Dataset):  # not in stat_mode
        output.save_to_disk(output_path)
        if args.clear_ds_cache:
            for cache_file in output_path.glob("cache-*.arrow"):
                os.remove(cache_file)
    else:
        with (output_path / "results.json").open("w") as f:
            json.dump(output, f, indent=4)
    with (output_path / "script_params.json").open("w") as f:
        json.dump(vars(args), f, indent=4)
    logger.info("Output saved to disk @ %s\n%s", output_path, "=" * 150)


if __name__ == "__main__":
    main()

