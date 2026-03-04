import os
import json
from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from torch import manual_seed
from torch.utils.data import DataLoader
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.models import Transformer, Pooling
from sentence_transformers.losses import CosineSimilarityLoss
from sentence_transformers.util import cos_sim

from partages_llm.utils import basic_logger_init


def parse_arguments():
    default_output_dir = os.path.join(os.getenv("HOME"), "partages-models/sts-encoders")
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("model_id", type=str)
    parser.add_argument("data_path", type=str)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=default_output_dir)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--nosave", action="store_true")
    return parser.parse_args()


def build_dataloader(df: pd.DataFrame, batch_size: int):
    dataset_list = list(
        InputExample(
            texts=[t.id_1, t.id_2], label=(t.sim / 5)
        ) for t in df.itertuples()
    )
    return DataLoader(dataset_list, batch_size=batch_size, shuffle=True)


def train_sts(
    model_id: str,
    seq_len: int,
    data_loader: DataLoader,
    epochs: int,
    lr: float,
    output_path: Optional[str] = None
):
    transformer = Transformer(model_id, max_seq_length=seq_len)
    model_input_names = ["input_ids", "attention_mask"]
    setattr(transformer.tokenizer, "model_input_names", model_input_names)
    pooling_layer = Pooling(
        transformer.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True
    )
    st_modules = transformer, pooling_layer
    model = SentenceTransformer(modules=st_modules)
    loss = CosineSimilarityLoss(model=model)
    model.fit(
        train_objectives=[(data_loader, loss)],
        epochs=epochs,
        scheduler="constantlr",
        optimizer_params={"lr": lr},
        weight_decay=.01,
        show_progress_bar=True,
        output_path=output_path,
    )
    return model


def main():
    args = parse_arguments()
    logger = basic_logger_init()

    logger.info("Loading dataset: %s", args.data_path)
    df = pd.read_csv(args.data_path, sep="\t")
    train_df = df[df.split == "train"]
    
    if not args.nosave:
        output_dir_name = Path(args.model_id).name + "_clister_" + datetime.now().strftime("%d-%m_%H-%M")
        output_dir_path = Path(args.output_dir) / output_dir_name
        output_dir_path.mkdir(parents=True)
        script_params = vars(args)
        script_params["file"] = __file__
        with (output_dir_path / "train_script_params.json").open("w") as f:
            json.dump(script_params, f, indent=4)
    else:
        output_dir_path = "n/a"

    manual_seed(args.seed)
    logger.info("Loading + launching model: %s", args.model_id)
    data_loader = build_dataloader(train_df, args.batch_size)
    output_path = None if (args.eval | args.nosave) else str(output_dir_path)
    model = train_sts(
        model_id=args.model_id,
        seq_len=args.seq_len,
        data_loader=data_loader,
        epochs=args.epochs,
        lr=args.lr,
        output_path=output_path
    )
    if args.eval:
        logger.info("Evaluating")
        eval_dataset = df[df.split == "test"].reset_index()
        cosine_similarities = np.fromiter(
            (cos_sim(*embeddings).item() for embeddings in zip(
                *map(
                    lambda x: model.encode(eval_dataset["id_" + str(x + 1)]),
                    range(2)
                )
            )),
            dtype=np.float64
        )
        result = spearmanr(eval_dataset.sim, cosine_similarities)
        logger.info("Spearman correlation for test data: %.5f", result.statistic)
        if not args.nosave:
            logger.info("Retraining on all data")
            data_loader = build_dataloader(df, args.batch_size)
            output_path = str(output_dir_path)
            _ = train_sts(
                model_id=args.model_id,
                seq_len=args.seq_len,
                data_loader=data_loader,
                epochs=args.epochs,
                lr=args.lr,
                output_path=output_path
            )
    logger.info("Done: output @ %s", output_dir_path)


if __name__ == "__main__":
    main()

