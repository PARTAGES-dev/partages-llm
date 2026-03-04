import os
import json
from pathlib import Path
from multiprocessing import cpu_count
from functools import partial
from argparse import ArgumentParser

from datasets import DatasetDict, load_from_disk, enable_progress_bars, disable_progress_bars
from jinja2 import Environment, FileSystemLoader, Template

from partages_llm.utils import basic_logger_init, make_version_subdir_path
from partages_llm.processing import instruction_to_prompt_completion


def get_template(template_path):
    jinja_env_dir, template_file_name = os.path.split(template_path)
    jinja_env = Environment(loader=FileSystemLoader(jinja_env_dir))
    return jinja_env.get_template(template_file_name)


def filter_func(x, source, split="validation"):
    sourcecheck = x["source"] == source
    splitcheck = x["source_split"] == split
    return sourcecheck and splitcheck


def _build_instruction(output, speciality, template):
    if isinstance(template, Template):  # MEDIQAL
        num_correct_answers = len(output.replace("\n", "").split(","))
        multiple_correct_answers = num_correct_answers > 1
        instruction = template.render(
            multiple_correct_answers=multiple_correct_answers,
            speciality=speciality
        )
    else:
        instruction = template
    return instruction


def build_instruction(output, speciality, template):
    instruction = _build_instruction(output, speciality, template)
    return {"instruction": instruction}


def build_instruction_batched(output, speciality, template):
    function = partial(_build_instruction, template=template)
    instructions = map(lambda x: function(*x), zip(output, speciality))
    return {"instruction": list(instructions)}


def sample_examples(
    ds,
    fewshot_template,
    num_examples,
    instruction_map_kwargs,
    seed,
    disable_tqdm=False
):
    if disable_tqdm:
        disable_progress_bars()
    ds_train_examples = ds.shuffle(seed=seed).take(num_examples).map(
        **instruction_map_kwargs
    )
    enable_progress_bars()
    question_list = [
        f"QUESTION {i + 1} : {q['question']}\nRÉPONSE {i + 1} : {q['output']}" \
            for i, q in enumerate(ds_train_examples)
    ]
    return fewshot_template.render(
        question_list=question_list,
        num_examples=num_examples
    )


def resample_wrap(instruction, **kwargs):
    fewshot_interstitial_text = sample_examples(**kwargs)
    return instruction_to_prompt_completion(
        instruction=instruction,
        interstitial_text=fewshot_interstitial_text
    )


def prompt_word_count_map_func(instance):
    return {"word_count": sum(len(turn["content"].split()) for turn in instance["prompt"])}


def prepare_and_write_output(data_dict, output_fp, wc_map_kwargs, max_wc, logger):
    dataset_dict = DatasetDict(data_dict).map(**wc_map_kwargs)
    if max_wc:
        dataset_dict = dataset_dict.filter(
            lambda x: x["word_count"] <= max_wc,
            desc=f"Removing prompts longer than {max_wc} words"
        )
    logger.info("OUTPUT DATASET:\n%s\n-> writing out to %s", repr(dataset_dict), output_fp)
    dataset_dict.save_to_disk(output_fp)


def main():
    logger = basic_logger_init()

    backitup = [os.pardir] * 3
    here = os.path.dirname(__file__)
    data_dir_base = os.path.normpath(os.path.join(here, *backitup, "data"))
    dataset_dir = Path(data_dir_base) / "wp2-instructions/preproc"
    default_template_path_instruction_dynamic = os.path.join(
        data_dir_base,
        "cfg/templates/mediqal-instructions-v0.jinja"
    )
    default_template_path_instruction_fixed = os.path.join(
        data_dir_base,
        "cfg/templates/fmcqa-instruction-v0.txt"
    )
    default_template_path_fewshot = os.path.join(
        data_dir_base,
        "cfg/templates/wp2-instructions-fewshot-interstitial.jinja"
    )
    default_num_workers = cpu_count() // 2

    parser = ArgumentParser()
    parser.add_argument("target_task_type", choices={"icl", "sft"})
    parser.add_argument("-v", "--dataset-version", type=int, default=0)
    parser.add_argument("-n", "--nshot", type=int, default=5)
    parser.add_argument("-m", "--max-word-count", type=int)
    parser.add_argument("-w", "--workers", type=int, default=default_num_workers)
    parser.add_argument("-b", "--map-bs", type=int, default=1)
    parser.add_argument(
        "-i", "--template-path-instruction-dynamic",
        default=default_template_path_instruction_dynamic
    )  # for MEDIQAL
    parser.add_argument(
        "-t", "--template-path-instruction-fixed",
        default=default_template_path_instruction_fixed
    )  # for FrenchMedMCQA
    parser.add_argument("-f", "--template-path-fewshot", default=default_template_path_fewshot)
    parser.add_argument("-r", "--resample", action="store_true")
    parser.add_argument("--seed", type=int, default=5318008)
    args = parser.parse_args()

    input_dir = dataset_dir / ("v" + str(args.dataset_version))
    logger.info("Loading dataset: %s", input_dir)
    ds = load_from_disk(input_dir)

    with open(args.template_path_instruction_fixed) as f:
        instruction_template_fixed = f.read()
    instruction_template_dynamic = get_template(args.template_path_instruction_dynamic)
    
    if args.target_task_type == "icl":
        fewshot_template = get_template(args.template_path_fewshot)
        suffix = f"-{args.nshot}shot"
    else:
        setattr(args, "resample", False)
        setattr(args, "nshot", None)
        suffix = ""

    if args.resample and args.workers > 1:
        logger.warning("Multiprocessing is not supported for example resampling; setting num_proc=0")
        setattr(args, "workers", 0)
    fmt_instruction_map_kwargs = {
        "desc": "Building instructions",
        "input_columns": ["output", "speciality"],
        "batched": args.map_bs > 1,
        "batch_size": args.map_bs,
    }
    fmt_prompt_map_kwargs = {
        "remove_columns": [
            "instruction",
            "speciality",
            "question",
            "output",
            "data_dir",
            "source",
            "source_split"
        ],
        "desc": "Building prompts",
        "num_proc": args.workers if args.workers > 1 else None
    }
    wc_map_kwargs = {
        "function": prompt_word_count_map_func,
        "desc": "Counting words"
    }
    out_f = partial(
        prepare_and_write_output,
        wc_map_kwargs=wc_map_kwargs,
        max_wc=args.max_word_count,
        logger=logger
    ) 
    source_name2instruction_template = {
        "MEDIQAL": instruction_template_dynamic,
        "FRENCHMEDMCQA": instruction_template_fixed
    }
    ds_fmt_output_dir = dataset_dir.parents[0] / "fmt" / args.target_task_type
    ds_fmt_output_fp = make_version_subdir_path(ds_fmt_output_dir, suffix=suffix, make=True)
    
    data_dict = {}
    for source, instruction_template in source_name2instruction_template.items():
        ff_eval = partial(filter_func, source=source)
        ds_eval_src = ds.filter(ff_eval, desc=source + " validation set")
        ff_train = partial(filter_func, source=source, split="train")
        ds_train_src = ds.filter(ff_train, desc="train examples")
        fmt_instruction_map_kwargs["function"] = partial(
            build_instruction_batched if fmt_instruction_map_kwargs["batched"] else build_instruction,
            template=instruction_template
        )
        if args.target_task_type == "sft":
            fmt_prompt_map_kwargs["function"] = instruction_to_prompt_completion
            ds_train_src_with_instructions = ds_train_src.map(**fmt_instruction_map_kwargs)
            ds_train_fmt = ds_train_src_with_instructions.map(**fmt_prompt_map_kwargs)
        elif args.resample:
            fmt_prompt_map_kwargs["function"] = partial(
                resample_wrap,
                ds=ds_train_src,
                fewshot_template=fewshot_template,
                num_examples=args.nshot,
                instruction_map_kwargs=fmt_instruction_map_kwargs,
                disable_tqdm=True,
                seed=args.seed
            )
        else:
            fewshot_interstitial_text = sample_examples(
                ds=ds_train_src,
                fewshot_template=fewshot_template,
                num_examples=args.nshot,
                instruction_map_kwargs=fmt_instruction_map_kwargs,
                seed=args.seed
            )
            fmt_prompt_map_kwargs["function"] = partial(
                instruction_to_prompt_completion,
                interstitial_text=fewshot_interstitial_text
            )
        ds_eval_src_with_instructions = ds_eval_src.map(**fmt_instruction_map_kwargs)
        ds_eval_fmt = ds_eval_src_with_instructions.map(**fmt_prompt_map_kwargs)
        if args.target_task_type == "icl":
            data_dict[source.lower()] = ds_eval_fmt
        else:
            out_f(
                data_dict={"train": ds_train_fmt, "validation": ds_eval_fmt},
                output_fp=ds_fmt_output_fp / source.lower()
            )
    if data_dict:
        out_f(data_dict=data_dict, output_fp=ds_fmt_output_fp)
    with (ds_fmt_output_fp / "script_params.json").open("w") as f:
        json.dump(vars(args), f, indent=4)


if __name__ == "__main__":
    main()

