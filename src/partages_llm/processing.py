import re
import warnings
from typing import Any, Callable, Dict, List, Optional, Union
from functools import partial
from dataclasses import dataclass

from datasets import Dataset, Features

TokenEncodingType = Dict[str, List[int]]


@dataclass
class ValidationSplitConfig:
    seed: int
    proportion: float
    num_validation_docs: Optional[int] = None


@dataclass
class DataMixConfig:
    # proportions relative to 1.0 for PARCOMED data
    transbio_proportion: float
    fineweb_proportion: float
    paradocs_proportion: float


def _feature_dict(
    dtype_str: str,
    ftype: str,
    names: Optional[List[str]] = None
) -> Dict[str, str]:
    ret = {
        "dtype": dtype_str,
        "_type": ftype
    }
    if names:
        ret["names"] = names
    return ret


def _feature_dict_seq(
    dtype_str: str,
    ftype: str="Value"
) -> Dict[str, Any]:
    dict_ = {
        "feature": _feature_dict(dtype_str, ftype),
        "_type": "Sequence"
    }
    return dict_


def get_tokenized_ds_features(class_label_names: Optional[List[str]] = None) -> Features:
    int8_feature = _feature_dict_seq("int8")
    features_dict = {
        "input_ids": _feature_dict_seq("int32"),
        "attention_mask": int8_feature,
        "special_tokens_mask": int8_feature
    }
    if class_label_names:
        features_dict["source"] = _feature_dict(
            "string", "ClassLabel", class_label_names
        )
    return Features.from_dict(features_dict)


def _enc_default_dict_init(
    bos_token_id: int,
    data: Optional[TokenEncodingType] = None
) -> TokenEncodingType:
    dict_ = {
        "input_ids": [bos_token_id],
        "attention_mask": [1],
        "special_tokens_mask": [1]
    }
    if data:
        for k, v in data.items():
            dict_[k].extend(v)
    return dict_


def _enc_len(*enx: TokenEncodingType) -> int:
    return sum(len(d["input_ids"]) for d in enx)


def _token_dict(iid_val: int, stm_val: int) -> Dict[str, int]:
    dict_ = {
        "input_ids": iid_val,
        "attention_mask": 1,
        "special_tokens_mask": stm_val
    }
    return dict_


def _generate_subsequences(
    enc: TokenEncodingType,
    sequence_length: int,
    stride: int,
    output_init_fn: Callable,
    eos_tokens: Optional[Dict[str, int]] = None
) -> Any:
    nonspec_input_token_ids = [
        iid for iid, stm in zip(enc['input_ids'], enc['special_tokens_mask']) if stm == 0
    ]
    total_nonspec_input_tokens = len(nonspec_input_token_ids)
    nonspec_token_block_size = sequence_length - (2 if eos_tokens else 1)
    for i, idx in enumerate(
        range(0, total_nonspec_input_tokens, nonspec_token_block_size)
    ):
        idx_lo = idx - (i * stride) + 1  # idx == i only when both are 0; +1 to skip existing bos_token
        idx_up = min(idx_lo + nonspec_token_block_size, total_nonspec_input_tokens + 1)
        subsequence_encoding_data = {}
        if idx_lo == idx_up:
            break
        for k, vals in enc.items():
            nonspec_token_block = vals[idx_lo:idx_up]
            if eos_tokens:
                nonspec_token_block.append(eos_tokens[k])
            subsequence_encoding_data[k] = nonspec_token_block
        yield output_init_fn(data=subsequence_encoding_data)


def _check_token_yield(
    dict_: TokenEncodingType,
    target_length: int,
    allow_lt: Optional[bool]=False
) -> TokenEncodingType:
    op = "<=" if allow_lt else "=="
    for encoding_key, token_list in dict_.items():
        assert isinstance(token_list, list), f"{encoding_key} not a list: {token_list}"
        length = len(token_list)
        assert eval(f"length {op} target_length"), f"|{encoding_key}| = {length}"
        if not all(isinstance(x, int) for x in token_list):
            raise RuntimeError(f"Not all integers: {encoding_key}:\n{token_list}")
    return dict_


def generate_concatenated_tokenized_ds(
    ds: Dataset,
    sequence_length: int,
    bos_token_id: int,
    space_id: int,
    eos_token_id: Optional[int] = None,
    stride: Optional[int] = None,
    return_remainder: Optional[bool] = True,
    minimum_remainder: Optional[int] = 5,
) -> TokenEncodingType:
    """
    Generator function intended for use with the `Dataset.from_generator` function.
    Takes a tokenized dataset and homogenises the lengths of it's sequences by
    concatenating them. Sequences are concatenated until their total length overflows
    the specified maximum, then truncated for the process to repeat.

    Args:
        ds: Input dataset - should at least have an `input_ids` feature.
        sequence_length: Target length for the token sequences.
        bos_token_id: Special token ID for the beginning of sequences.
        space_id: Token ID for the space character - this is required for more sensible
            truncation.
        eos_token_id: Special token ID for the end of sequences.
        stride: The size of the overlap to allow between truncated sections of the same
            input sequence.
        return_remainder: Whether to return the final sequence in the dataset when it is
            shorter than the target length
        minimum_remainder: Minimum size to consider for the aforementioned final sequence.
    
    Yields:
        Concatenated token sequences of the specified length.
    """
    output_encoding = partial(_enc_default_dict_init, bos_token_id=bos_token_id)
    output_check = partial(_check_token_yield, target_length=sequence_length)
    intermediate_tokens = _token_dict(space_id, 0)
    accumulated_encodings = output_encoding()
    gen_kwargs = {
        "sequence_length": sequence_length,
        "stride": stride,
        "output_init_fn": output_encoding
    }
    if eos_token_id is not None:
        gen_kwargs["eos_tokens"] = _token_dict(eos_token_id, 1)
    gen_subseq = partial(_generate_subsequences, **gen_kwargs)
    for encoding in ds:
        # add encoding to accumulation
        if encoding["input_ids"][-1] != eos_token_id:
            encoding_tmp = encoding
        else:
            # this will no longer be the end of a sequence so we don't include the eos_token
            for k in encoding.keys():
                encoding_tmp[k].extend(encoding[k][1:-1])
        
        if encoding_tmp["input_ids"][-1] != space_id:
            # sequences shouldn't be concatenated directly; we put a space at the end if
            # there isn't already one there
            for k, val in intermediate_tokens.items():
                encoding_tmp[k].append(val)
        for k in encoding.keys():
            # the function output_encoding takes care of the bos_token so we always exclude it
            accumulated_encodings[k].extend(encoding_tmp[k][1:])
        
        if _enc_len(accumulated_encodings) < sequence_length:
            # not enough tokens to yield, just move on
            continue
        else:
            for subseq in gen_subseq(accumulated_encodings):
                try:
                    yield output_check(subseq)
                except AssertionError:
                    # may happen on the last iteration if the last subsequence has less
                    # than sequence_length tokens; in this case the subsequence is retained
                    # for the next iteration
                    accumulated_encodings = subseq
    if return_remainder and _enc_len(accumulated_encodings) > minimum_remainder:
        for subseq in gen_subseq(accumulated_encodings):
            yield output_check(subseq, allow_lt=True)


def instruction_to_prompt_completion(
    instruction: Union[str, Dict[str, str]], 
    interstitial_text: str = "",
    question: Optional[str] = None,
    output: Optional[str] = None
) -> Dict[str, List[Dict[str, str]]]:
    if isinstance(instruction, str):
        system_content = instruction
        if question is None or output is None:
            in_type = type(instruction)
            err_msg = f"No `question` or `output` attributes provided for prompt construction; type(instruction)={in_type}"
            raise ValueError(err_msg)
    else:
        # column mapping
        system_content = instruction["instruction"]
        question = instruction["question"]
        output = instruction["output"]
    formatted_instance = {
        "prompt": [{
            "role": "system",
            "content": system_content + interstitial_text
        }, {
            "role": "user",
            "content": question
        }],
        "completion": [{
            "role": "assistant",
            "content": output
        }]
    }
    return formatted_instance


def get_mcq_answer_pattern(dataset: Dataset) -> re.Pattern:
    """
    Builds a regex pattern based on the answer keys in the `completion` column of the dataset.
    """
    letters = dataset.map(
        lambda x: {'letters': re.sub(r'[^A-Z]', '', x['completion'][0]['content'])},
        desc="Extracting MCQ answer options",
    )['letters']
    answer_options = ''.join(set(''.join(letters)))
    return re.compile(fr'[,\.\s>]?[{answer_options}][,\.\s<]')


def infer_answer_split_tokens_for_text_generation(
    dataset: Dataset,
    original_col: str,
    templated_col: str,
    idx: int
) -> str:
    templated_prompt = dataset[templated_col][idx]
    original_final_prompt_element = dataset[original_col][idx][-1]["content"]
    final_prompt_element_template_idx = templated_prompt.find(original_final_prompt_element)
    if final_prompt_element_template_idx == -1:
        warnings.warn("User content not found in original prompt when trying to infer generation prompt")
        return
    answer_split_idx = final_prompt_element_template_idx + len(original_final_prompt_element)
    split_token_s = templated_prompt[answer_split_idx:]
    templated_col_endswith_split_token = map(lambda s: s.endswith(split_token_s), dataset[templated_col])
    assert sum(templated_col_endswith_split_token) == len(dataset), \
        f"Not all prompts end with the inferred split token {split_token_s}"
    return split_token_s


# regex patterns for cleaning - putting them here so they'll be compiled only once on import which I
# would imagine might slightly speed up large-scale mapping applications of the function
PUNCT_ASCII = r"!-/:-@[-`{-~'\."
PUNCT_ASCII_PATTERN = re.compile(rf"[{PUNCT_ASCII}]")
LETTERS = r"A-Za-zŒœÀ-ÂÇ-ÊÏÔà-âç-êïôû"
CHAR_EXCLUDE_PATTERN = re.compile(rf"[^{LETTERS}0-9{PUNCT_ASCII}\s]")
PUNCT_DEDUP_PATTERN = re.compile(rf"([{PUNCT_ASCII}\s])\1+")
PUNCT_INSERT_PATTERN_A = re.compile(r"([:;,!\?\.])(?=\S)")
PUNCT_INSERT_PATTERN_B = re.compile(r"(?<=\S)([:;!\?])")
LONGWORDS_PATTERN = re.compile(r"\s?\w{35,}" + rf"[{PUNCT_ASCII}]?\s?")  # can't use f-strings when there are {}s that we want regex to parse


def _word_check_pattern(n: int) -> re.Pattern:
    return re.compile(r"^(?=(?:.*\b[{" + LETTERS + r"}]+\b){" + str(n) + "}).*$")


def matches_word_check(text: str, n: int) -> bool:
    # ignoring punctuation and line breaks, the text must have at least n words that are made up only of letters
    ptrn = _word_check_pattern(n)
    text_check = re.sub(PUNCT_ASCII_PATTERN, "", text.replace("\n", " "))
    return bool(re.match(ptrn, text_check))


def clean_text(text: str, strict: Optional[bool]=None, word_check_min: int=3) -> str:
    # homogenise apostrophes
    text = re.sub("’", "'", text)

    if strict:
        # if there aren't "enough" words, returns an empty string that'll be removed later in preprocessing
        if not matches_word_check(text, word_check_min):
            return ""

    # remove any superfluous characters
    text = re.sub(CHAR_EXCLUDE_PATTERN, "", text)

    # remove duplicate spaces/punctuation
    text = re.sub(PUNCT_DEDUP_PATTERN, r"\1", text)

    # insert spaces after punctuation if necessary
    text = re.sub(PUNCT_INSERT_PATTERN_A, r"\1 ", text)

    # insert spaces before punctuation if necessary (positive lookbehind for non-space characters)
    text = re.sub(PUNCT_INSERT_PATTERN_B, r" \1", text)

    # remove abnormally long (>=35 character) words - most likely errors
    # https://fr.wikipedia.org/wiki/Liste_des_mots_les_plus_longs_en_fran%C3%A7ais
    text = re.sub(LONGWORDS_PATTERN, "", text)
    return text.strip()

