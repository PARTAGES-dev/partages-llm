import os
import json
import tempfile
import subprocess
import multiprocessing as mp
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm
from lxml import etree
from uuid import uuid4
from functools import partial
from itertools import product, repeat

from datasets import Dataset, DatasetDict, load_from_disk

from partages_llm.utils import basic_logger_init, make_version_subdir_path


def parse_arguments():
    default_output_dir = os.path.normpath(os.path.join(
        os.path.dirname(__file__), *[os.pardir] * 3,
        "data/wp2-corpus/"
    ))
    parser = ArgumentParser()
    parser.add_argument("corpus_path", type=str)
    parser.add_argument("-n", dest="ngram", type=int, nargs="*", default=3)
    parser.add_argument("-t", dest="threshold", type=float, nargs="*", default=.5)
    parser.add_argument("-b", dest="buffer", type=int, default=16777216 * 4)
    parser.add_argument("-l", dest="doc_limit", type=int, default=int(1e9))
    parser.add_argument("-m", dest="max_sentences_per_vert_doc", type=int, default=100)
    parser.add_argument("-q", dest="silent", action="store_true")
    parser.add_argument("-v", dest="save_input_vert_file", action="store_true")
    parser.add_argument("-s", dest="shuffle_corpus", action="store_true")
    parser.add_argument("-S", dest="stat_mode", action="store_true")
    parser.add_argument("-c", dest="clear_ds_cache", action="store_true")
    parser.add_argument("-w", dest="max_workers", type=int, default=1)
    parser.add_argument("-o", dest="output_dir", type=str, default=default_output_dir)
    return parser.parse_args()


def vert_doc_template(doc_id, name, content):
    doc = f'\n<doc id="{doc_id}" name="{name}">'
    if isinstance(content, list):
        for p in content:
            doc += f'\n<p>\n{"\n".join(p.split())}\n</p>'
    elif isinstance(content, str):
        doc += f'\n<p>\n{"\n".join(content.split())}\n</p>'
    doc += "\n</doc>"
    return doc


def get_counts_vert(f_io):
    n_docs = n_tokens = n_bytes = 0
    for line in f_io:
        if line.startswith("<doc id="):
            n_docs += 1
        elif not line.startswith("<"):
            n_tokens += 1
            n_bytes += len(line.encode("utf-8"))
    return {"docs": n_docs, "tokens": n_tokens, "bytes": n_bytes}


def run_onion(input_path, exec_, opts, output_dir, logger=None, log_prefix=""):
    output_path = output_dir / "corpus-dedup.vert"
    dedup_cmd = [exec_, *opts.split(), str(input_path)]
    logger.info(
        "%sRunning deduplication command: %s > %s",
        log_prefix, " ".join(dedup_cmd), output_path
    )
    returncode = subprocess.call(dedup_cmd, stdout=output_path.open("w"))
    logger.info("%sReturn code %d", log_prefix, returncode)
    return output_path


def statmode_func(ngram, threshold, buffer, run_kwargs):
    opts = f"-qsmn {ngram} -t {threshold} -b {buffer}"
    output_filepath = run_onion(opts=opts, **run_kwargs)
    dedup_output_counts = get_counts_vert(output_filepath.open())
    dedup_output_counts.update({"ngram": ngram, "threshold": threshold})
    return dedup_output_counts


def statmode_func_mp_wrapper(proc_offset, ngram, threshold, buffer, tmp_path, run_kwargs):
    proc = mp.current_process()._identity[0] + proc_offset
    output_dir = tmp_path / f"output-{proc}"
    output_dir.mkdir()
    run_kwargs.update({
        "log_prefix": f"Process {proc}: ",
        "output_dir": output_dir
    })
    return statmode_func(ngram, threshold, buffer, run_kwargs)


def vert2xml(f_io, root_name="corpus", disable_pb=False, total_lines=None):
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


def main(args):
    log_level = "error" if args.silent else "info"
    logger = basic_logger_init(log_level)
    if os.path.isdir(args.corpus_path):
        logger.info("Loading data from %s", args.corpus_path)
        ds = load_from_disk(args.corpus_path)
        if isinstance(ds, DatasetDict):
            ds = ds["train"]
        if args.shuffle_corpus:
            ds = ds.shuffle(seed=0)
        vert_file_path_input = None
    elif os.path.isfile(args.corpus_path) and args.corpus_path.endswith(".vert"):
        vert_file_path_input = Path(args.corpus_path)
    logger.info("Dataset loaded:\n%s", repr(ds))
    tempfile.tempdir = (Path(__file__).parents[0] / "../../../data/tmp").resolve()
    tempfile.tempdir.mkdir(exist_ok=True)
    onion_exec = os.getenv("HOME") + "/onion"
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir.decode())
        if vert_file_path_input is None:
            vert_file_loc_input = Path(args.corpus_path) if args.save_input_vert_file else tmp_path
            vert_file_path_input = vert_file_loc_input / "corpus.vert"
            tqdm_total = min(ds.num_rows, args.doc_limit)
            tqdm_desc = f"Building input file {vert_file_path_input.name}"
            def template_wo(f_io, doc, idx, content):
                source = doc.get("source", "partages-wp2")
                subset = doc.get("subset", "train")
                name_elems = f"doc{idx}", source, subset, uuid4().hex
                name = "__".join(name_elems)
                f_io.write(vert_doc_template(idx, name, content))
                return idx + 1
            with vert_file_path_input.open("w") as f:
                i = 0
                for doc in tqdm(ds, disable=args.silent, total=tqdm_total, desc=tqdm_desc):
                    if not doc["text"]:
                        continue
                    if i >= args.doc_limit:
                        break
                    if doc["text"].count(". ") >= 2:
                        text = [sentence + "." for sentence in doc["text"].split(". ")]
                        n_sent = len(text)
                        if n_sent > args.max_sentences_per_vert_doc:
                            for j in range(0, n_sent, args.max_sentences_per_vert_doc):
                                k = min(j + args.max_sentences_per_vert_doc, n_sent)
                                i = template_wo(f, doc, i, text[j:k])
                        else:
                            i = template_wo(f, doc, i, text)
                    else:
                        i = template_wo(f, doc, i, doc["text"])
        input_st_size = vert_file_path_input.stat().st_size
        logger.info("Input file size: documents %d, bytes %d", i, input_st_size)
        if args.stat_mode:
            output = {
                "original": get_counts_vert(vert_file_path_input.open()),
                "dedup_runs": []
            }
            run_onion_kwargs = {
                "input_path": vert_file_path_input,
                "exec_": onion_exec,
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
                        proc_offset = i * num_proc
                        map_args = list((j, *t) for j, t in zip(repeat(proc_offset), params_this_iter))
                        output["dedup_runs"].extend(pool.starmap(map_func, map_args))
            else:
                run_onion_kwargs["output_dir"] = tmp_path
                for ngram, threshold in product(args.ngram, args.threshold):
                    output["dedup_runs"].append(
                        statmode_func(ngram, threshold, args.buffer, run_onion_kwargs)
                    )
        else:
            onion_opts = f"-smn {args.ngram[0]} -t {args.threshold[0]} -b {args.buffer}"
            if args.silent:
                onion_opts += " -q"
            vert_file_path_output = run_onion(
                vert_file_path_input, onion_exec, onion_opts, tmp_path, logger=logger
            )
            output_st_size = vert_file_path_output.stat().st_size
            percent_shrinkage = 100 - (100 * output_st_size / input_st_size)
            logger.info(
                "Output file %s size in bytes: %d - shrunk by %.1f%%",
                vert_file_path_output.name, output_st_size, percent_shrinkage
            )
            total_lines = int(subprocess.check_output(["wc", "-l", vert_file_path_output]).split()[0])
            with vert_file_path_output.open("r") as f:
                xml_lines = vert2xml(f, disable_pb=args.silent, total_lines=total_lines)
            xml_path = vert_file_path_output.parents[0] / vert_file_path_output.name.replace("vert", "xml")
            logger.info("Writing to %s...", xml_path)
            with xml_path.open("w") as f:
                f.write("".join(xml_lines))
            logger.info("Generating Dataset from XML parser...")
            def etree_gen():
                for document in etree.parse(xml_path).findall("doc"):
                    metadata = document.attrib["name"].split("__")
                    yield {
                        "source": metadata[1],
                        "subset": metadata[2],
                        "doc_id": metadata[3],
                        "text": " ".join(p.text for p in document.findall("p") if p.text)
                    }
            try:
                output = Dataset.from_generator(etree_gen)
            except Exception as exc:
                logger.error("%s encountered: dumping vert & XML files...", type(exc).__name__)
                for p in (vert_file_path_output, xml_path):
                    os.rename(p, tempfile.tempdir / "dedup-debug-dump" / p.name)
                raise exc
        logger.info("Discarding temporary directory...")
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
    main(parse_arguments())

