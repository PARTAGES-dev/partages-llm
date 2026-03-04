import json
from datetime import datetime
from random import randint
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path

import torch
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM

from partages_llm.utils import basic_logger_init
from partages_llm.processing import get_mcq_answer_pattern, infer_answer_split_tokens_for_text_generation
from partages_llm.eval.mcqa import mcqa


def main():

    ## VARS SETUP ##
    dataset_names = "frenchmedmcqa", "mediqal"
    logger = basic_logger_init()
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("model_path")
    parser.add_argument("data_dir")
    parser.add_argument("-v", "--dataset-version", type=int, default=0)
    parser.add_argument("-d", "--dataset-name", choices=dataset_names, default=dataset_names[0])
    parser.add_argument("--ndocs", type=int)
    parser.add_argument("--max-gen-tokens", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pad-token")
    parser.add_argument("--ct", dest="chat_template")
    parser.add_argument("-t", dest="temperature", type=float, default=1.)
    parser.add_argument("-p", dest="sampling_top_p", type=float, default=.9)
    parser.add_argument("-o", dest="write_out", action="store_true")
    parser.add_argument("-a", dest="write_out_all", action="store_true")
    parser.add_argument("--db", dest="debug_mode", action="store_true")
    parser.add_argument("--peft", action="store_true")
    args = parser.parse_args()

    ## INPUT DATA ##
    version_prefix = "v" + str(args.dataset_version)
    data_dir_base = Path(args.data_dir)
    formatting_run = list(data_dir_base.glob(version_prefix + "*/")).pop()
    data_path = data_dir_base / formatting_run / args.dataset_name
    assert data_path.exists(), f"Tried to find {data_path} and couldn't"
    logger.info("Loading data from %s", data_path)
    eval_dataset_text = datasets.load_from_disk(data_path)
    if args.ndocs:
        eval_dataset_text = eval_dataset_text.shuffle(seed=0).take(args.ndocs)
    
    ## MODEL SETUP ##
    logger.info("Loading model")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.peft:
        model = AutoPeftModelForCausalLM.from_pretrained(
            args.model_path,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            ignore_mismatched_sizes=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            dtype=torch.bfloat16
        )
    logger.info("Loaded: %s", type(model).__name__)
    model = model.to(device)

    ## PREPROCESSING ##
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left")
    if tokenizer.chat_template is None and args.chat_template is not None:
        with open(args.chat_template) as f:
            setattr(tokenizer, "chat_template", f.read())
    if not tokenizer.pad_token:
        if args.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
        else:
            tokenizer.add_special_tokens({"pad_token": args.pad_token})
    logger.info("Processing prompts")
    eval_dataset_templated = eval_dataset_text.map(
        lambda x: {
            "templated_prompt": tokenizer.apply_chat_template(
                x["prompt"],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
        },
        desc="Applying chat template"
    )
    split_token_s = infer_answer_split_tokens_for_text_generation(
        dataset=eval_dataset_templated,
        original_col="prompt",
        templated_col="templated_prompt",
        idx=randint(0, len(eval_dataset_templated))
    )
    eval_dataset_tokenized_input = eval_dataset_templated.map(
        lambda x: tokenizer(x["templated_prompt"]),
        remove_columns=["prompt"],
        desc="Tokenizing"
    )
    sorted_dataset = eval_dataset_tokenized_input.map(
        lambda x: {"seq_len": len(x["input_ids"])},
        desc="Counting tokens",
    ).sort("seq_len", reverse=False)
    logger.info("Max. sequence length %d", max(sorted_dataset["seq_len"]))

    ## LFG ##
    mcq_answer_pattern = get_mcq_answer_pattern(eval_dataset_text)
    return_all_outputs = args.write_out and args.write_out_all
    logger.info("Launching generations")
    if split_token_s is not None:
        logger.info("Generation prompt marker:\n%s", split_token_s)
    result = mcqa(
        model,
        tokenizer,
        dataset=sorted_dataset,
        top_p=args.sampling_top_p,
        batch_size=args.batch_size,
        temperature=args.temperature,
        answer_split_token=split_token_s,
        max_new_tokens=args.max_gen_tokens,
        return_all_outputs=return_all_outputs,
        inspect_responses_live=args.debug_mode,
        mcq_answer_pattern=mcq_answer_pattern
    )
    t = "\n\t\t\t\t - "
    metric_disp_str = t.join(
        f"{k.upper()} = {round(v * 100, 4)}" for k, v in result["metrics"].items()
    )
    logger.info("EVAL SET METRICS:%s%s", t, metric_disp_str)

    ## DOCUMENT ##
    if args.write_out:
        results_dir_path = data_dir_base / "eval-results/icl"
        timestamp = datetime.now().strftime("%y-%m-%d_%H-%M")
        output_dirname = "-".join((Path(args.model_path).name, args.dataset_name, timestamp))
        output_path = results_dir_path / output_dirname
        output_path.mkdir(parents=True)
        arg_dict = vars(args)
        arg_dict["input_dataset_full_path"] = str(data_path)
        with (output_path / "script_params.json").open("w") as f:
            json.dump(arg_dict, f, indent=4)
        with (output_path / "results.json").open("w") as f:
            json.dump(result, f, indent=4)
        logger.info("Results saved @ %s", output_path)
    print("=" * 75)


if __name__ == "__main__":
    main()

