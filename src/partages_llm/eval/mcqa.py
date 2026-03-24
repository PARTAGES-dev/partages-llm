import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from tqdm import tqdm
from collections import defaultdict

from datasets import Dataset
from peft import PeftModelForCausalLM
from transformers import BatchEncoding, PreTrainedModel, PreTrainedTokenizerFast


def _prepare_dataset(dataset: Dataset, batch_size: int) -> Dataset:
    if "seq_len" in dataset.features:
        dataset_to_batch = dataset
    else:
        dataset_to_batch = dataset.map(
            lambda x: {"seq_len": len(x["input_ids"])},
            desc="Counting tokens",
        ).sort("seq_len", reverse=False)
    prepared_dataset = dataset_to_batch.batch(batch_size)
    return prepared_dataset


def _prepare_batch(
    batch: Dict[str, List[int]],
    keys: List[str],
    tokenizer: PreTrainedTokenizerFast,
    device: str
) -> BatchEncoding:
    batch_model_input = {k: v for k, v in batch.items() if k in keys}
    tensors = tokenizer.pad(
        batch_model_input, "longest", "left", return_tensors="pt"
    ).to(device)
    return tensors


def _generate_response_text(
    model: Union[PreTrainedModel, PeftModelForCausalLM],
    batch_tensors: BatchEncoding,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    tokenizer: PreTrainedTokenizerFast
) -> List[str]:
    do_sample = temperature > 0.
    outputs = model.generate(
        **batch_tensors,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id
    )
    input_sequence_dim = batch_tensors.input_ids.shape[1]
    output_ids = outputs[:, input_sequence_dim:].detach().cpu().numpy()
    text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return text


def _prepare_answer_labels(
    generated_text: str,
    mcq_answer_pattern: re.Pattern,
    tokenizer: PreTrainedTokenizerFast,
    label: List[Dict[str, str]]
) -> Tuple[Set[str]]:
    answer_matches = re.findall(mcq_answer_pattern, generated_text)
    answer_set = set(map(lambda s: re.sub('[^A-Z]', '', s), answer_matches))
    target = label[0]["content"].strip("\n")
    y_set = set(target.split(","))
    return answer_set, y_set


def _calculate_metric_inputs(answer_set: Set[str], y_set: Set[str]) -> Tuple[int]:
    if answer_set:
        overlap = answer_set.intersection(y_set)
        correct = len(overlap)  # true positives
        incorrect = len(answer_set - overlap)  # false positives
        missed = len(y_set - overlap)  # false negatives
        exact = int(correct == len(y_set))
    else:
        correct = incorrect = exact = 0
        missed = len(y_set)
    return correct, incorrect, missed, exact


def _calculate_metrics(counts: Dict[str, int]) -> Dict[str, float]:
    metrics = {}
    metrics["accuracy"] = counts["exact_match"] / counts["total_docs"]
    precision_denom = counts["num_correct_responses"] + counts["num_incorrect_responses"]
    precision = (counts["num_correct_responses"] / precision_denom) if precision_denom else 0.
    metrics["precision"] = precision
    recall_denom = counts["num_correct_responses"] + counts["num_missed_responses"]
    recall = (counts["num_correct_responses"] / recall_denom) if recall_denom else 0.
    metrics["recall"] = recall
    f1_denom = precision + recall
    metrics["f1"] = (2 * precision * recall / f1_denom) if f1_denom else 0.
    return metrics


def mcqa(
    model: Union[PreTrainedModel, PeftModelForCausalLM],
    tokenizer: PreTrainedTokenizerFast,
    dataset: Dataset,
    batch_size: int,
    max_new_tokens: int,
    mcq_answer_pattern: re.Pattern,
    answer_split_token: Optional[str] = None,
    temperature: float = .05,
    top_p: float = .9,
    inspect_responses_live: bool = False,
    return_all_outputs: bool = False,
) -> Dict[str, Any]:
    
    ## SETUP ##
    generation_input_keys = "input_ids", "attention_mask"
    progress_desc = "Running Generation + MCQ Evaluation"
    if not tokenizer.pad_token_id:
        # a padding token is needed to prepare homogenous batches
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    eval_counts = defaultdict(int)  # to hold metric inputs
    eval_counts["total_docs"] += len(dataset)
    if return_all_outputs:
        # containers for full run details
        generations = {}
        num_batch_tokens = []    

    ## DATASET PREP ##
    batched_dataset = _prepare_dataset(dataset, batch_size)
    batch_iterator = tqdm(batched_dataset, desc=progress_desc, disable=inspect_responses_live)
    
    for batch_idx, batch in enumerate(batch_iterator):
        
        ## INPUT COLLATION ##
        batch_tensors = _prepare_batch(batch, generation_input_keys, tokenizer, model.device)
        if return_all_outputs:
            # keep track of how many total tokens are being passed in
            # this can sometimes help with diagnosing memory issues etc.
            batch_dim = batch_tensors["input_ids"].shape
            num_batch_tokens.append(batch_dim[0] * batch_dim[1])
        
        ## INFERENCE ##
        output_text_list = _generate_response_text(
            model, batch_tensors, temperature, top_p, max_new_tokens, tokenizer
        )
        evaluation_data = zip(batch["doc_id"], output_text_list, batch["completion"])
        
        ## RAW SCORES ##
        for i, (id_, generated_text, label) in enumerate(evaluation_data):
            answer_set, y_set = _prepare_answer_labels(
                generated_text, mcq_answer_pattern, tokenizer, label
            )
            if inspect_responses_live:
                # prints out details and waits for confirmation before continuing
                # allows you to read the model's answers as they're generated
                print(f"DOC : {id_} ({batch_idx + i}/{eval_counts['total_docs']})")
                print("LABEL:", ", ".join(y_set))
                print("GENERATED TEXT :", generated_text)
                print("EXTRACTED ANSWER :", ", ".join(answer_set))
                input("Press Enter to continue")
            metric_inputs = _calculate_metric_inputs(answer_set, y_set)
            eval_counts["num_correct_responses"] += metric_inputs[0]
            eval_counts["num_incorrect_responses"] += metric_inputs[1]
            eval_counts["num_missed_responses"] += metric_inputs[2]
            eval_counts["exact_match"] += metric_inputs[3]
            if return_all_outputs:
                generations[id_] = generated_text
    
    ## METRICS ##
    metrics = _calculate_metrics(eval_counts)
    ret = {"metrics": metrics}
    if return_all_outputs:
        ret["generations"] = generations
        ret["eval_counts"] = eval_counts
        ret["num_batch_tokens"] = num_batch_tokens
    
    return ret

