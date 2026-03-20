import os
import gc
import json
import warnings
import traceback
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path
from typing import Any, Dict, Union
from datetime import datetime
from uuid import uuid4
from itertools import product

import torch
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, Gemma3Model, set_seed
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

from partages_llm.utils import basic_logger_init, ignored
from partages_llm.processing import get_mcq_answer_pattern
from partages_llm.eval.mcqa import mcqa

_DATASET_NAMES = "frenchmedmcqa", "mediqal"


def parse_arguments():
    home = os.getenv("HOME")
    default_output_dir = os.path.join(home, "partages-models/sft-runs/")
    default_data_dir = os.path.join(home, "partages-data/sft")
    lora_init_options = "eva", "pissa_niter_16", "id"

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("model_path")
    parser.add_argument("-v", "--dataset-version", type=int, default=0)
    parser.add_argument("-r", "--rank-dimension", type=int, default=16)
    parser.add_argument("-n", "--ndocs", type=int)
    parser.add_argument("-o", "--output-dir", default=default_output_dir)
    parser.add_argument("-d", "--dataset-name", choices=_DATASET_NAMES)
    parser.add_argument("--ctp", dest="chat_template_path")
    parser.add_argument("--lora-init", choices=lora_init_options, default="id")
    parser.add_argument("--use-dora", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--tt", dest="target_tokens", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--eval-steps", type=float, default=.1)
    parser.add_argument("--log-steps", type=int, default=100)
    parser.add_argument("--batch-size", dest="train_batch_size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--grad-acc", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--schedule", default="constant")
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--lora-dropout", type=float, default=.05)
    parser.add_argument("--wu", dest="warmup", type=float, default=.15)
    parser.add_argument("--mml", dest="model_max_length", type=int, default=2048)
    parser.add_argument("--eval-acc", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--opt", default="adamw_torch_fused")
    parser.add_argument("--dd", dest="ds_dir", default=default_data_dir)
    parser.add_argument("--pad-token")
    parser.add_argument("--hps-cfg")
    parser.add_argument("--hps-dev-frac", type=float, default=.05)
    parser.add_argument("--max-hps-iter", type=int, default=500)
    parser.add_argument("--interactive-bs", action="store_true")
    parser.add_argument("--ll", dest="log_level", default="info")
    return parser.parse_args()


def load_model(model_path, pad_token=None):
    base_model = AutoModelForCausalLM.from_pretrained(model_path)
    if isinstance(base_model, Gemma3Model):
        base_model = base_model.language_model  # we only want the text model
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    if not tokenizer.pad_token:
        if pad_token is None:
            tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
        else:
            tokenizer.add_special_tokens({"pad_token": pad_token})
    base_model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    return base_model, tokenizer


def load_datasets():
    ds_dir = Path(args.ds_dir) / ("v" + str(args.dataset_version))
    assert ds_dir.exists(), f"{ds_dir} not found"
    if args.dataset_name:
        ds = datasets.load_from_disk(ds_dir / args.dataset_name)
        train_ds = ds["train"]
        eval_ds = ds["validation"]
    else:
        ds_load_dict = {fp.name: datasets.load_from_disk(fp) for fp in ds_dir.glob("*/")}
        assert set(ds_load_dict) == set(_DATASET_NAMES), f"Unexpected dataset name found in {list(ds_load_dict)}"
        train_ds = datasets.concatenate_datasets(
            [ds_["train"] for ds_ in ds_load_dict.values()]
        ).shuffle(seed=args.seed)
        eval_ds = datasets.DatasetDict({
            name: ds_["validation"] for name, ds_ in ds_load_dict.items()
        })
    if args.ndocs:
        train_ds = train_ds.shuffle(seed=args.seed).take(args.ndocs)
    logger.info("Train dataset loaded: %s", repr(train_ds))
    return train_ds, eval_ds


def run_training(
    model,
    tokenizer,
    train_ds,
    eval_ds=None,
    output_dir=None,
    logging_dir=None,
    save_strategy="epoch"
):
    if args.target_tokens:
        modules_to_save = None
        outputs = set(map(lambda x: x[0]["content"].replace("\n", ""), train_ds["completion"]))  # set of answers
        target_tokens = set("".join(outputs))  # set of unique tokens appearing in the answers 
        target_token_ids = [x.pop() for x in tokenizer(list(target_tokens))["input_ids"]]
        logger.info("Targeted tokens: `%s`\n\t\t\tIDs: %s", "`  `".join(target_tokens), ", ".join(map(str, target_token_ids)))
        trainable_token_indices = {"embed_tokens": target_token_ids}
    else:
        modules_to_save = ["lm_head", "embed_tokens"]
        trainable_token_indices = None
    logger.info("Setting up LORA training configuration")
    init_lora_weights = True if args.lora_init == "id" else args.lora_init
    peft_config = LoraConfig(
        r=args.rank_dimension,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules="all-linear",
        modules_to_save=modules_to_save,
        task_type="CAUSAL_LM",
        use_dora=args.use_dora,
        use_rslora=True,
        trainable_token_indices=trainable_token_indices,
        init_lora_weights=init_lora_weights
    )
    logger.info("PEFT Config:\n%s", repr(peft_config))
    if output_dir is None:
        model_basename = Path(args.model_path).name
        timestamp = datetime.now().strftime("%y-%m-%d_%H-%M")
        run_id = os.environ.get("SLURM_JOB_ID", uuid4().hex)
        output_basename = f"{model_basename}_sft_{timestamp}-{run_id}"
        output_dir = Path(args.output_dir) / output_basename
        output_dir.mkdir(parents=True)
    if logging_dir is None:
        logging_dir = output_dir / "logs"
    eval_kwargs = {} if eval_ds is None else {
        "per_device_eval_batch_size": args.eval_batch_size,
        "eval_strategy": "steps",
        "eval_steps": args.eval_steps,
        "eval_accumulation_steps": args.eval_acc,
        "batch_eval_metrics": True
    }
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.grad_acc,
        max_length=args.model_max_length,
        dataset_kwargs={
            "add_special_tokens": False,  # special tokens handled by template
            "append_concat_token": False,
        },
        group_by_length=True,
        chat_template_path=args.chat_template_path,
        optim=args.opt,
        learning_rate=args.lr,
        max_grad_norm=.3,
        warmup_ratio=args.warmup,
        lr_scheduler_type=args.schedule,
        completion_only_loss=True,
        save_strategy=save_strategy,
        bf16=True,
        logging_steps=args.log_steps,
        logging_dir=logging_dir,
        push_to_hub=False,
        torch_empty_cache_steps=args.eval_acc,
        seed=args.seed,
        **eval_kwargs
    )
    peft_model = get_peft_model(model, peft_config)
    trainer = SFTTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    logger.info("*** Launching training ***")
    trainer.train()
    return trainer


def run_hps_iter(
    iter_idx: int,
    output_dir_path: Path,
    hp_kwargs: Dict[str, Any],
    ds: datasets.Dataset,
    bs_adjustments: int,
) -> Union[SFTTrainer, int, type(None)]:
    """
    Wraps `run_training` with some extra stuff for managing multiple runs where different parameter
    combinations are being tested.

    Returns:
        The Trainer object for the run, if it succeeds. If interactive batch size adjustment mode is
        switched on and training fails because of an OOM error, the function will return a counter for
        the number of adjustments made to the batch size during this run
    """
    run_subdir_path = output_dir_path / f"run{iter_idx}-logs"
    hp_kwargs_disp = ", ".join(f"{k}={v}" for k, v in hp_kwargs.items())
    logger.info("Setting variable hyperparameters for run %d: %s", iter_idx, hp_kwargs_disp)
    args.__dict__.update(hp_kwargs)
    base_model, tokenizer = load_model(args.model_path, args.pad_token)
    try:
        trainer = run_training(
            model=base_model,
            tokenizer=tokenizer,
            train_ds=ds["train"],
            eval_ds=ds["test"],
            logging_dir=run_subdir_path,
            output_dir=run_subdir_path,
            save_strategy="no",
        )
    except Exception as exc:
        if isinstance(exc, torch.OutOfMemoryError) and args.interactive_bs:
            # command line input parsing to set a new batch size
            cl_in_t = input("\n\n** CUDA OOM ** Batch size to modify [t/e]: ")  # choose train_batch_size or eval_batch_size
            new_batch_size_type = cl_in_t.strip()
            arg_to_update = f"{'train' if new_batch_size_type == 't' else 'eval'}_batch_size"
            cl_in_v = input("New smaller batch size: ")
            new_batch_size = int(cl_in_v.strip())
            setattr(args, arg_to_update, new_batch_size)
            if new_batch_size <= 0:
                logger.error("Invalid value for new batch size")
                raise exc

            ## memory cleanup ##
            torch.cuda.empty_cache()
            gc.collect()

            return bs_adjustments + 1  # does not retry iteration
        traceback.print_exc()
        logger.error("Training failed, saving current hyperparameter search results + quitting [=> %s]", output_dir_path)
        return
    return trainer


def main():

    ## VARS SETUP ##
    global args, logger
    args = parse_arguments()
    if args.eval_batch_size is None:
        setattr(args, "eval_batch_size", args.train_batch_size)
    if args.eval_steps >= 1.:
        setattr(args, "eval_steps", int(args.eval_steps))

    logger = basic_logger_init(args.log_level)
    set_seed(args.seed)
    warnings.simplefilter("ignore", UserWarning)
    arg_dict = vars(args)
    nt = "\n" + "\t" * 3
    if args.hps_cfg:
        with Path(args.hps_cfg).open() as f:
            hyperparameter_search_config = json.load(f)
        for hp in hyperparameter_search_config:
            with ignored(KeyError):
                del arg_dict[hp]
    arg_str = nt.join(f"{k}: {v}" for k, v in arg_dict.items())
    logger.info("Parameters:%s%s", nt, arg_str)
    logger.info("Num. GPUs: %d", torch.cuda.device_count())
    
    ## INPUT DATA ##
    train_ds, eval_ds = load_datasets()
        
    if args.hps_cfg:
        
        ## HYPERPARAMETER SEARCH SETUP ##
        hps_output_basename = f"gs_{Path(args.model_path).name}_" + datetime.now().strftime("%y-%m-%d_%H-%M")
        output_dir_path = Path(args.output_dir) / hps_output_basename
        output_dir_path.mkdir()
        with (output_dir_path / "script_params.json").open("w") as f:
            json.dump(vars(args), f, indent=4)
        hps_results = []
        hpsc_processed = {k: [True, False] if v is None else v for k, v in hyperparameter_search_config.items()}
        dev_split_ds = train_ds.train_test_split(test_size=args.hps_dev_frac, shuffle=True, seed=args.seed)
        logger.info("* HPS dataset *\n%s", repr(dev_split_ds))
        bs_adjustments = 0
        intermediate_txt_file_path = output_dir_path / "hps-results-intermediate.txt"
        
        for i, value_combination in enumerate(product(*hpsc_processed.values())):
            if (i - bs_adjustments) == args.max_hps_iter:
                logger.warning("Reached iteration limit of %d; stopping", args.max_hps_iter)
                break
            
            hp_kwargs = dict(zip(hpsc_processed, value_combination))
            trainer = run_hps_iter(i + 1, output_dir_path, hp_kwargs, dev_split_ds, bs_adjustments)
            if isinstance(trainer, int):
                continue
            elif trainer is None:
                with (output_dir_path / "hps-results.jsonl").open("w") as f:
                    json.dump(hps_results, f, indent=4)
                exit(1)
            trainer_metrics = trainer.evaluate()
            eval_metrics = mcqa(
                model=trainer.model.merge_and_unload().eval(),
                tokenizer=trainer.tokenizer,
                dataset=trainer.eval_dataset,
                batch_size=args.eval_batch_size,
                max_new_tokens=64,
                mcq_answer_pattern=get_mcq_answer_pattern(trainer.eval_dataset),
            )
            run_data = {
                "iter": i,
                "params": hp_kwargs,
                "trainer_metrics": trainer_metrics,
                "trainer_logs_in": trainer.args.output_dir.name,
                "eval_metrics": eval_metrics
            }
            with intermediate_txt_file_path.open("a" if i else "w") as f:
                f.write(repr(run_data) + "\n")
            hps_results.append(run_data)
            mcqa_results_disp = ", ".join(f"{k}={round(v, 4)}" for k, v in eval_metrics["metrics"].items())
            logger.info("Run %d finished\nResults: %s\n%s", i + 1, mcqa_results_disp, "-" * 100)
            logger.debug("CUDA Memory Summary:\n%s", torch.cuda.memory_summary())
            
            ## memory cleanup ##
            trainer.model.delete_adapter("default")
            trainer.model.zero_grad(set_to_none=True)
            trainer.model.cpu()
            for attr in ("model", "optimizer", "train_dataset", "eval_dataset", "lr_scheduler"):
                setattr(trainer, attr, None)
            del trainer
            torch.cuda.empty_cache()
            gc.collect()
            logger.debug("CUDA Allocated memory after cleanup: %.2f MB", torch.cuda.memory_allocated() / 1024 ** 2)
        
        with (output_dir_path / "hps-results.jsonl").open("w") as f:
            json.dump(hps_results, f, indent=4)
        logger.info("Results saved @ %s\n%s", output_dir_path, "=" * 100)
    else:
        model, tokenizer = load_model(args.model_path, args.pad_token)
        trainer = run_training(model, tokenizer, train_ds)
        
        logger.info("Merging LoRA and base models")
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(
            trainer.args.output_dir,
            safe_serialization=True,
            max_shard_size="1GB",
            save_embedding_layers=True
        )
        tokenizer.save_pretrained(trainer.args.output_dir)
        logger.info("Merged model saved @ %s", trainer.args.output_dir)
        
        with (trainer.args.output_dir / "script_params.json").open("w") as f:
            json.dump(arg_dict, f, indent=4)
        
        ## EVAL ##
        if not args.skip_eval:
            logger.info("Evaluating on validation set")
            if tokenizer.chat_template is None and args.chat_template is not None:
                with open(args.chat_template) as f:
                    setattr(tokenizer, "chat_template", f.read())
            eval_ds = eval_ds.map(
                lambda x: tokenizer.apply_chat_template(
                    x["prompt"],
                    add_generation_prompt=True,
                    enable_thinking=False,
                    return_dict=True,
                    padding=False  # this'll be done during batching in evaluation func
                ),
                desc="Applying chat template + tokenizing eval dataset",
                remove_columns=["prompt"],
            )
            def _metric_disp_str(metric_dict):
                return "\n\t".join(f"{k.upper()} = {round(v * 100, 2)}" for k, v in metric_dict.items())
            if isinstance(eval_ds, datasets.DatasetDict):
                # run eval on both datasets
                eval_results = {}
                for dataset_name in eval_ds:
                    eval_results_iter = mcqa(
                        model=merged_model,
                        tokenizer=tokenizer,
                        dataset=eval_ds[dataset_name],
                        mcq_answer_pattern=get_mcq_answer_pattern(eval_ds[dataset_name]),
                        batch_size=args.eval_batch_size,
                        max_new_tokens=16
                    )
                    logger.info("%s EVAL SET METRICS:\n\t%s", dataset_name.upper(), _metric_disp_str(eval_results_iter["metrics"]))
                    eval_results[dataset_name] = eval_results_iter
            else:
                eval_results = mcqa(
                    model=merged_model,
                    tokenizer=tokenizer,
                    dataset=eval_ds,
                    mcq_answer_pattern=get_mcq_answer_pattern(eval_ds),
                    batch_size=args.eval_batch_size,
                    max_new_tokens=16
                )
                logger.info("%s EVAL SET METRICS:\n\t%s", args.dataset_name.upper(), _metric_disp_str(eval_results["metrics"]))
            with (trainer.args.output_dir / "eval-results.json").open("w") as f:
                json.dump(eval_results, f, indent=4)


if __name__ == "__main__":
    main()

