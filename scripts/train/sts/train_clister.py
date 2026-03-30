import json
from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from torch import manual_seed
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.models import Transformer, Pooling
from sentence_transformers.losses import CosineSimilarityLoss
from sentence_transformers.util import cos_sim
from scipy.stats import spearmanr

from partages_llm.utils import basic_logger_init

DESC="Script for fine-tuning a BERT model on the CLISTER sentence similarity dataset."
DATAPATH_HELP = "Path to the input .tsv file"
SEQLEN_HELP = "Sequence length"
LR_HELP = "Learning rate"
SKIPEVAL_HELP = "Skip the test set evaluation - training only"
RETRAIN_HELP = "After test evaluation, continue the model's training on the held-out data. "\
"This should only be done for models that will be used for other downstream and tasks and "\
"not evaluated on CLISTER."


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description=DESC, formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("model_id", type=str)
    parser.add_argument("data_path", type=str, help=DATAPATH_HELP)
    parser.add_argument("--seq-len", type=int, default=512, help=SEQLEN_HELP)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5, help=LR_HELP)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--skip-eval", action="store_true", help=SKIPEVAL_HELP)
    parser.add_argument("--retrain", action="store_true", help=RETRAIN_HELP)
    return parser.parse_args()


def build_train_dataloader(df: pd.DataFrame, batch_size: int) -> DataLoader:
    dataset_list = [
        InputExample(
            texts=[t.id_1, t.id_2],  # pair of input sentences
            label=(t.sim / 5)  # similarity score: originally in [0-5]; map to (0,1) for evaluation
        ) for t in df.itertuples()  # iterates over the rows of the dataset
    ]
    return DataLoader(dataset_list, batch_size=batch_size, shuffle=True)


def train_sts(
    model_id: str,
    seq_len: int,
    data_loader: DataLoader,
    epochs: int,
    lr: float,
    output_path: Optional[str] = None
) -> SentenceTransformer:
    # first we load the underlying transformer encoder
    transformer = Transformer(model_id, max_seq_length=seq_len)
    model_input_names = ["input_ids", "attention_mask"]
    setattr(transformer.tokenizer, "model_input_names", model_input_names)
    
    # to fine-tune the above word-embedding model as a sentence transformer,
    # we add a pooling module
    pooling_layer = Pooling(
        transformer.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True
    )
    st_modules = transformer, pooling_layer
    model = SentenceTransformer(modules=st_modules)
    
    # organize training parameters and launch
    loss = CosineSimilarityLoss(model=model)
    train_objectives = [(data_loader, loss)]
    optimizer_params = {"lr": lr}
    model.fit(
        train_objectives=train_objectives,
        optimizer_params=optimizer_params,
        output_path=output_path,
        scheduler="constantlr",
        show_progress_bar=True,
        weight_decay=.01,
        epochs=epochs,
    )
    return model


def main():

    ## VARS SETUP ##
    args = parse_arguments()
    logger = basic_logger_init()
    manual_seed(args.seed)
    if args.output_dir is None:
        output_dir_path = "n/a"
    else:
        output_dir_name = Path(args.model_id).name + "_clister_" + datetime.now().strftime("%d-%m_%H-%M")
        output_dir_path = Path(args.output_dir) / output_dir_name
        output_dir_path.mkdir(parents=True)
        script_params = vars(args)
        script_params["file"] = __file__
        with (output_dir_path / "train_script_params.json").open("w") as f:
            json.dump(script_params, f, indent=4)
    output_path = str(output_dir_path) if (args.skip_eval or not args.retrain) else None

    ## INPUT DATA ##
    logger.info("Loading dataset: %s", args.data_path)
    df = pd.read_csv(args.data_path, sep="\t")
    train_df = df[df.split == "train"]
    data_loader = build_train_dataloader(train_df, args.batch_size)
    
    ## RUN TRAINING ##
    logger.info("Loading + launching model: %s", args.model_id) 
    model = train_sts(
        data_loader=data_loader,
        output_path=output_path,
        model_id=args.model_id,
        seq_len=args.seq_len,
        epochs=args.epochs,
        lr=args.lr,
    )

    if not args.skip_eval:
        ## EVAL ##
        logger.info("Evaluating")
        eval_dataset = df[df.split == "test"].reset_index()
        
        # this function embeds the sentence pairs id_1 and id_2 
        data_to_embedding_map = map(lambda x: model.encode(eval_dataset["id_" + str(x + 1)]), range(2))

        # calculate the cosine similarity for each pair
        embedding_pairs_to_similarities_iter = (cos_sim(*embeddings).item() for embeddings in zip(*data_to_embedding_map))
        cosine_similarities = np.fromiter(embedding_pairs_to_similarities_iter, dtype=np.float64)
        
        # compare the ground-truth similarities ("sim" attribute of the dataset) to the calculated ones
        result = spearmanr(eval_dataset.sim, cosine_similarities)
        logger.info("Spearman correlation for test data: %.5f", result.statistic)
        
        if args.retrain:
            logger.info("Retraining on all data")
            data_loader = build_train_dataloader(df, args.batch_size)
            output_path = str(output_dir_path)
            _ = train_sts(
                data_loader=data_loader,
                output_path=output_path,
                model_id=args.model_id,
                seq_len=args.seq_len,
                epochs=args.epochs,
                lr=args.lr,
            )
    
    logger.info("Done: output @ %s\n%s", output_dir_path, "=" * 75)


if __name__ == "__main__":
    main()

