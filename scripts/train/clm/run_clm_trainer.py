import os
import sys
import json
import traceback
from datetime import datetime
from uuid import uuid4
from functools import partial
from pathlib import Path
from argparse import ArgumentParser

import torch
import datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from transformers.utils import logging as hf_logging

global IDRIS  # flag: are we on Jean Zay?
try:
    # need this module to run multinode training on Jean Zay
    # see https://github.com/idriscnrs/idr_torch 
    import idr_torch
except ModuleNotFoundError:
    IDRIS = False
else:
    IDRIS = True

from partages_llm.utils import basic_logger_init
from partages_llm.training_tools import ProfilerCallback

PREC2DTYPE = {"bf": torch.bfloat16, "fp": torch.float16, "no": torch.float}
TIMESTAMP = datetime.now().strftime("%y-%m-%d-%H-%M")
SCHEDULERS = {"constant", "linear", "cosine"}


def parse_arguments():
    base_dir_variable = "WORK" if IDRIS else "HOME"
    base_dir = os.getenv(base_dir_variable)
    default_output_dir = os.path.join(base_dir, "partages-models/clm-runs")
    parser = ArgumentParser()
    parser.add_argument("model_path", type=str)
    parser.add_argument("data_path", type=str)
    parser.add_argument("--eval_data_path", type=str)
    parser.add_argument("--eval_split_name", type=str, default="val")
    parser.add_argument("--output_dir", type=str, default=default_output_dir)
    parser.add_argument("--pad_token", type=str, default="<pad>")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_acc", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=97531)
    parser.add_argument("--steps", type=int, default=-1)  # replicate TrainingArguments default
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--log_steps", type=int, default=100)
    parser.add_argument("--fsdp-cfg", dest="fsdp_config_path", type=str)
    parser.add_argument("--prec", type=str, choices=tuple(PREC2DTYPE), default="no")
    parser.add_argument("--opt", type=str, default="adamw_torch")
    parser.add_argument("--sched", type=str, choices=SCHEDULERS, default="constant")
    parser.add_argument("--pb", action="store_true")
    parser.add_argument("--gcp", dest="gradient_checkpointing", action="store_true")
    parser.add_argument("--acp", dest="activation_checkpointing", action="store_true")
    parser.add_argument("--prof", action="store_true")
    parser.add_argument("--no_eval", action="store_true")
    return parser.parse_args()


def ndnt(k: int):
    """
    Helper function for logger readability; shortcut for indenting lines

    Returns: str
    """
    return "\n" + "\t" * k


def get_torch_rank():
    """
    Return the current process rank 

    Returns: str
    """
    if IDRIS:
        return idr_torch.rank
    return int(os.environ.get("RANK", "0"))


def get_torch_local_rank():
    """
    Return the current process rank local to the node
    (i.e. identify the current GPU among the other GPUs on
    the same node to which this job has access)

    Returns: str
    """
    if IDRIS:
        return idr_torch.local_rank
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_local_main_process():
    """
    Is the current process the main process for this node?

    Returns: bool
    """
    return get_torch_local_rank() == 0


def is_main_process():
    """
    Is the current process the overall main process for this job?

    Returns: bool
    """
    return get_torch_rank() == 0


def setup_training_arguments(args, logger):
    try:
        use_fsdp = os.path.isfile(args.fsdp_config_path)
    except TypeError:
        use_fsdp = False
    fsdp_cfg = None
    if use_fsdp:
        fsdp_cfg_str = ""
        with Path(args.fsdp_config_path).open() as f:
            fsdp_cfg = json.load(f)
        fsdp_cfg["activation_checkpointing"] = args.activation_checkpointing
        if args.gradient_checkpointing:
            logger.info("[FSDP] Turning off gradient checkpointing")
            setattr(args, "gradient_checkpointing", False)  # https://github.com/huggingface/transformers/issues/30404
        for kw, data in fsdp_cfg.items():
            if isinstance(data, dict):
                for k, v in data.items():
                    fsdp_cfg_str += ndnt(4) + f"{k}: {v}"
            else:
                fsdp_cfg_str += ndnt(3) + f"{kw}: {data}"
        logger.info("* FSDP Configuration: *%s", fsdp_cfg_str)
    if not args.pb:
        hf_logging.disable_progress_bar()
    model_path = Path(args.model_path)
    if model_path.name.startswith("checkpoint"):
        # for continuing training from an existing CLM checkpoint
        model_basename = model_path.parents[0].name.split("_")[0] + "_cp"
    else:
        model_basename = model_path.name
    job_id = os.environ.get("SLURM_JOB_ID", uuid4().hex)
    output_basename = model_basename + "_" + TIMESTAMP + "-" + job_id
    output_dir = Path(args.output_dir) / output_basename
    logging_dir = output_dir / "logs"
    if is_main_process():
        logging_dir.mkdir(parents=True)
        with (output_dir / "script_params.json").open("w") as f:
            json.dump(vars(args), f, indent=4)
    eval_strategy = "no" if args.no_eval else "steps"
    use_bf16 = args.prec == "bf"
    use_fp16 = args.prec == "fp"
    dataloader_num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
    local_rank = get_torch_local_rank()
    disable_tqdm = not args.pb
    return TrainingArguments(
        output_dir=output_dir,
        eval_strategy=eval_strategy,
        eval_steps=args.log_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        gradient_checkpointing=args.gradient_checkpointing,
        learning_rate=args.lr,
        weight_decay=.05,
        num_train_epochs=args.epochs,
        max_steps=args.steps,
        optim=args.opt,
        lr_scheduler_type=args.sched,
        warmup_ratio=.0,  # better for CPT I think
        save_steps=args.save_steps,
        logging_steps=args.log_steps,
        logging_dir=logging_dir,
        seed=args.seed,
        bf16=use_bf16,
        fp16=use_fp16,
        dataloader_num_workers=dataloader_num_workers,
        local_rank=local_rank,
        disable_tqdm=disable_tqdm,
        fsdp=use_fsdp,
        fsdp_config=fsdp_cfg,
        save_only_model=True,
        report_to="tensorboard",
    )


def load_datasets():
    logger.info("Loading dataset from %s", args.data_path)
    tokenized_ds = datasets.load_from_disk(args.data_path)
    logger.info("Dataset loaded: %d rows", tokenized_ds.num_rows)
    tokenized_ds_eval = None
    if args.eval_data_path:
        logger.info("Loading evaluation dataset from %s", args.eval_data_path)
        try:
            tokenized_ds_eval = datasets.load_from_disk(args.eval_data_path)
        except FileNotFoundError:
            logger.warning("Evaluation dataset %s not found; turning off evaluation step", args.eval_data_path)
            setattr(args, "no_eval", True)
        else:
            logger.info("Evaluation dataset loaded: %d rows", tokenized_ds_eval.num_rows)
    elif not args.no_eval:
        try:
            tokenized_ds_eval = tokenized_ds[args.eval_split_name]
        except KeyError:
            logger.warning("Evaluation split `%s` not found in dataset; turning off evaluation step", args.eval_split_name)
            setattr(args, "no_eval", True)
    if isinstance(tokenized_ds, datasets.DatasetDict):
        tokenized_ds = tokenized_ds["train"]
    logger.info("Training dataset: \n%s", repr(tokenized_ds))
    if tokenized_ds_eval is not None:
        logger.info("Evaluation dataset: \n%s", repr(tokenized_ds_eval))
    return tokenized_ds, tokenized_ds_eval


def run_training(train_args, train_ds, eval_ds):
    logger.info("Initialising model + tokenizer")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        local_files_only=IDRIS
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.add_special_tokens({"pad_token": args.pad_token})
    
    logger.info("Initialising trainer")
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, seed=args.seed)
    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=train_args,
        data_collator=data_collator
    )

    logger.info("*** Launching training ***")
    hf_logging.set_verbosity(train_args.get_process_log_level())
    hf_logging.enable_default_handler()
    hf_logging.enable_explicit_format()
    try:
        if args.prof:
            profile_trace_dir = trainer.args.output_dir / "profiles"
            if is_main_process():
                profile_trace_dir.mkdir()
            def profile_trace_handler(profiler, d):
                with (d / f"trace{profiler.step_num}.txt").open("w") as f:
                    f.write(
                        profiler.key_averages().table(
                            sort_by="self_cuda_time_total", row_limit=-1
                        )
                    )
            profiler_schedule = torch.profiler.schedule(
                wait=5, warmup=1, active=5
            )
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA
                ],
                on_trace_ready=partial(profile_trace_handler, d=profile_trace_dir),
                schedule=profiler_schedule,
                profile_memory=True,
                with_stack=True,
                record_shapes=True
            ) as profiler:
                trainer.add_callback(ProfilerCallback(profiler))
                train_result = trainer.train()
        else:
            train_result = trainer.train()
    except Exception:
        rank = get_torch_rank()
        logger.error("TRAINER FAILED @ PROCESS %d", rank)  # just so there'll be a timestamp in the stdout for when a process fails
        tb = traceback.format_exc()
        print(tb.replace("\n", f"\n[rank{rank}] "), file=sys.stderr)
        torch.distributed.destroy_process_group()
        return

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_model(trainer.args.output_dir)
    logger.info("Done! Output @ %s\n%s", trainer.args.output_dir, "=" * 75)


def main():
    ## DISTRIBUTED GPU SETUP ##
    world_size = idr_torch.world_size if IDRIS else -1
    rank = get_torch_rank()
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )

    ## VARS SETUP ##
    global logger, args
    args = parse_arguments()
    log_level = "info" if is_main_process() else "error"
    logger = basic_logger_init(log_level)
    
    bookend = "=" * 5
    logger.info("%s   Causal Language Modelling: Continued Pretraining Run   %s", bookend, bookend)
    arg_str = ndnt(3).join(f"{k}: {v}" for k, v in args.__dict__.items())
    logger.info("* Parameters *%s%s", ndnt(3), arg_str)
    gpu_name = torch.cuda.get_device_properties(0).name
    if IDRIS:
        node_list = idr_torch.nodelist
        num_nodes = len(node_list)
        linebreak_text = ndnt(3)
    else:
        world_size = torch.distributed.get_world_size()  # can only be run post-initialisation
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
        num_nodes = world_size // local_world_size
        linebreak_text = "[Note: number of compute nodes was estimated based on world_size/local_world_size, may be inaccurate]" + ndnt(3)
    ebs = world_size * args.batch_size * args.grad_acc
    logger.info(
        "%d processes (%d compute nodes w/ %s)%s=> Effective Batch Size = %d",
        world_size, num_nodes, gpu_name, linebreak_text, ebs
    )

    ## LAUNCH ##
    datasets = load_datasets()
    train_args = setup_training_arguments()
    run_training(train_args, *datasets)


if __name__ == "__main__":
    main()

