import re
from typing import Dict, Optional, Union
from tqdm import tqdm
from collections import defaultdict

from datasets import Dataset
from peft import PeftModelForCausalLM
from transformers import PreTrainedModel, PreTrainedTokenizerFast


def mcqa(
    model: Union[PreTrainedModel, PeftModelForCausalLM],
    tokenizer: PreTrainedTokenizerFast,
    dataset: Dataset,
    batch_size: int,
    max_new_tokens: int,
    mcq_answer_pattern: re.Pattern,
    answer_split_token: Optional[str] = None,
    temperature: float = 1.,
    top_p: float = .9,
    inspect_responses_live: bool = False,
    return_all_outputs: bool = False,
) -> Dict[str, float]:
    eval_counts = defaultdict(int)
    if not tokenizer.pad_token_id:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    total_docs = len(dataset)
    if not "seq_len" in dataset.features:
        dataset = dataset.map(
            lambda x: {"seq_len": len(x["input_ids"])},
            desc="Counting tokens",
        ).sort("seq_len", reverse=False)
    batched_dataset = dataset.batch(batch_size)
    if return_all_outputs:
        generations = {}
        num_batch_tokens = []
    generation_input_keys = "input_ids", "attention_mask"
    for batch_idx, batch in enumerate(tqdm(
        batched_dataset,
        desc="Running generation+eval",
        disable=inspect_responses_live
    )):
        batch_model_input = {k: v for k, v in batch.items() if k in generation_input_keys}
        batch_tensors = tokenizer.pad(
            batch_model_input, "longest", "left", return_tensors="pt"
        ).to(model.device)
        if return_all_outputs:
            batch_dim = batch_tensors["input_ids"].shape
            num_batch_tokens.append(batch_dim[0] * batch_dim[1])
        outputs = model.generate(
            **batch_tensors,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id
        )
        output_ids = outputs.detach().cpu().numpy()
        output_text_list = tokenizer.batch_decode(output_ids)
        for i, (id_, str_, label) in enumerate(
            zip(batch["doc_id"], output_text_list, batch["completion"])
        ):
            if answer_split_token is None:
                input_text = tokenizer.decode(batch_tensors["input_ids"][i])
                answer_split_idx = len(input_text)
                answer = str_[answer_split_idx:]
            else:
                answer = str_.split(answer_split_token)[-1]
            answer_match = re.match(mcq_answer_pattern, answer)
            answer_clean = answer_match.group(1) if answer_match else ''
            if inspect_responses_live:
                print(f"DOC : {id_} ({batch_idx + i}/{total_docs})")
                print("GEN :", answer)
                print("ANSWER :", answer_clean + "\n")
            target = label[0]["content"].strip("\n")
            y_set = set(target.split(","))
            if answer_clean:
                x_set = set(answer_clean.split(","))
                overlap = x_set.intersection(y_set)
                num_correct_responses = len(overlap)  # true positives
                num_incorrect_responses = len(x_set - overlap)  # false positives
                num_missed_responses = len(y_set - overlap)  # false negatives
                exact_match = int(num_correct_responses == len(y_set))
            else:
                num_correct_responses = num_incorrect_responses = exact_match = 0
                num_missed_responses = len(y_set)
            eval_counts["num_correct_responses"] += num_correct_responses
            eval_counts["num_incorrect_responses"] += num_incorrect_responses
            eval_counts["num_missed_responses"] += num_missed_responses
            eval_counts["exact_match"] += exact_match
            if return_all_outputs:
                generations[id_] = str_
    metrics = {}
    metrics["accuracy"] = eval_counts["exact_match"] / len(dataset)
    precision_denom = eval_counts["num_correct_responses"] + eval_counts["num_incorrect_responses"]
    precision = (eval_counts["num_correct_responses"] / precision_denom) if precision_denom else 0.
    metrics["precision"] = precision
    recall_denom = eval_counts["num_correct_responses"] + eval_counts["num_missed_responses"]
    recall = (eval_counts["num_correct_responses"] / recall_denom) if recall_denom else 0.
    metrics["recall"] = recall
    f1_denom = precision + recall
    metrics["f1"] = (2 * precision * recall / f1_denom) if f1_denom else 0.
    ret = {"metrics": metrics}
    if return_all_outputs:
        ret["generations"] = generations
        ret["eval_counts"] = eval_counts
        ret["num_batch_tokens"] = num_batch_tokens
    return ret

