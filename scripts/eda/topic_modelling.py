import os
import json
import argparse
import warnings
import traceback
from pathlib import Path
from typing import Dict, List, Union

import hdbscan
import datasets
import numpy as np
import umap.umap_ as umap
from jinja2 import Environment, FileSystemLoader
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from stop_words import get_stop_words

from partages_llm.utils import basic_logger_init, make_version_subdir_path

_DATADIR_BASE = Path(os.getenv("HOME")) / "partages-llm-data"
DESC="Topic modelling pipeline inspired by BERTopic (https://github.com/MaartenGr/BERTopic). "\
"The input dataset is embedded using a SentenceTransformer model, before the dimensions are reduced using UMAP. "\
"These reduced embeddings are clustered using HDBSCAN, then representative keywords are extracted from these clusters using TF-IDF. "\
"The final output of this script is a prompt intended for an LLM to assign semantic labels to the clusters."
MODEL_HELP = "Identifier for the Sentence Transformer BERT model."
DATA_HELP = "Name of the dataset to load. This can be a Hugging Face dataset - either a local path or a Hub ID - "\
"or a previously-embedded dataset of vectors, to be loaded with `numpy.load`."
NDOCS_HELP = "Number of documents to encode and cluster: if not specified, all docs in the dataset will be used."
TEXTCOL_HELP = "Dataset column from which to extract documents for embedding+clustering."
BS_HELP = "Batch size to use for encoder inference."
UMAP_NC_HELP = "Number of dimensions for UMAP dimensionality reduction. Typically between 2 and 10."
UMAP_NN_HELP = "Neighbours for UMAP's local neighbourhood approximation. \
Larger values capture more global structure, smaller ones more local structure."
HDBS_MCS_HELP = "Minimum number of samples in a cluster for HDBSCAN. \
Smaller values yield more clusters, larger values fewer but denser ones."
DIST_HELP="Distance metric to use for clustering."
TFIDF_K_HELP = "Number of top keywords to extract per cluster using c-TF-IDF. \
These keywords will be used in LLM prompts."
PROMPT_N_HELP = "Number of sample documents to include in each LLM prompt for context."
TEMPLATE_HELP = "Path to a Jinja template file to organise the clusters."
OUT_HELP = "Directory in which to save the output."
VRB_HELP = "Enable verbose output, printing summary statistics after each major pipeline step."

# scikit-learn built-in pairwise distances
PAIRWISE={"cityblock", "cosine", "euclidean", "l1", "l2", "manhattan", "nan_euclidean"}
EXCLUDE_SOURCES = ['WMT16']


def parse_arguments():
    default_output_dir = os.path.join(_DATADIR_BASE, "topic-modelling")
    default_prompt_template_file = os.path.join(_DATADIR_BASE, "config/templates/topic-modelling-prompt-v0.jinja")
    parser = argparse.ArgumentParser(
        description=DESC, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("bert_model_id", type=str, help=MODEL_HELP)
    parser.add_argument("dataset_id", type=str, help=DATA_HELP)
    parser.add_argument("-e", "--save_embeddings", action="store_true")
    parser.add_argument("-d", "--n_docs", type=int, help=NDOCS_HELP)
    parser.add_argument("-t", "--text_column_name", type=str, default="text", help=TEXTCOL_HELP)
    parser.add_argument("-b", "--batch_size", type=int, default=32, help=BS_HELP)
    parser.add_argument("-c", "--umap_n_components", type=int, default=5, help=UMAP_NC_HELP)
    parser.add_argument("-u", "--umap_n_neighbours", type=int, default=15, help=UMAP_NN_HELP)
    parser.add_argument("-H", "--hdbscan_min_cluster_size", type=int, default=100, help=HDBS_MCS_HELP)
    parser.add_argument("-D", "--distance_metric", type=str, default="euclidean", choices=PAIRWISE, help=DIST_HELP)
    parser.add_argument("-k", "--ctfidf_top_k", type=int, default=10, help=TFIDF_K_HELP)
    parser.add_argument("-n", "--sample_n_docs", type=int, default=3, help=PROMPT_N_HELP)
    parser.add_argument("-p", "--prompt_template_file", type=str, default=default_prompt_template_file, help=TEMPLATE_HELP)
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("-o", "--output_dir", type=str, default=default_output_dir, help=OUT_HELP)
    parser.add_argument("-v", "--verbose", action="store_true", help=VRB_HELP)
    parser.add_argument("-G", "--use_gpu", action="store_true")
    return parser.parse_args()


def embed_documents(
    documents: List[str],
    bert_model_name: str,
    batch_size: int,
    device: Union[str, List[str]] = "cpu",
    pb: bool = False
) -> np.ndarray:
    """
    Embeds a list of documents using a pre-trained Sentence Transformer model.

    Args:
        documents: A list of strings, where each string is a document.
        bert_model_name: Identifier for the Sentence Transformer model (e.g., "all-MiniLM-L6-v2").
        pb: If True, show progress bar for the encoding step.

    Returns:
        A numpy array where each row is the embedding for a corresponding document.
    """
    model = SentenceTransformer(bert_model_name)
    return model.encode(
        documents, 
        batch_size=batch_size,
        device=device,
        show_progress_bar=pb,
        convert_to_numpy=True
    )


def save_embeddings(
    embeddings: np.ndarray,
    dir_path: Path,
    document_ids: List[str],
    metadata: Dict[str, str]
) -> Path:
    """
    Saves an array of document embeddings to disk along with the document IDs from the
    original input dataset (array is saved in the compressed .npz format).

    Args:
        embeddings: The embedding array to save.
        dir_path: The directory in which to save them.
        document_ids: A list of document IDs in the corresponding order to `embeddings`.
        metadata: Details about the embeddings to be saved; will be written to an adjacent JSON file.
    
    Returns:
        A pathlib.Path object corresponding to the location of the saved embeddings.
    """
    id2embedding = dict(zip(document_ids, embeddings))
    save_path = make_version_subdir_path(dir_path, make=True, stem="st")
    np.savez_compressed(save_path / "embeddings", **id2embedding)
    with (save_path / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=4)
    return save_path


def reduce_dimensions(
    embeddings: np.ndarray,
    n_components: int = 5,
    n_neighbours: int = 15,
    random_state: int = 42,
    verbose: bool = False
) -> np.ndarray:
    """
    Reduces the dimensionality of document embeddings using UMAP (Uniform Manifold Approximation and Projection).

    Args:
        embeddings: A numpy array of document embeddings.
        n_components: The number of dimensions to reduce the embeddings to.
        n_neighbors: The size of local neighborhood (in terms of number of neighboring
                     sample points) used for manifold approximation.
        random_state: Seed for reproducibility of UMAP projections.
        verbose: Passed to UMAP.

    Returns:
        A numpy array of reduced-dimension embeddings.
    """
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbours,
        min_dist=0.,  # Recommended by BERTopic for better cluster separation
        metric="cosine",  # Cosine distance is often effective for embeddings
        random_state=random_state,
        verbose=verbose,  # UMAP has its own verbose flag
        tqdm_kwds={"disable": True}  # the default behaviour with verbose=True is to print text updates AND tqdm: it's ugly 
    )
    return reducer.fit_transform(embeddings)


def perform_clustering(
    reduced_embeddings: np.ndarray,
    min_cluster_size: int = 10,
    distance_metric: str = "euclidean"
) -> np.ndarray:
    """
    Performs HDBSCAN clustering on reduced-dimension embeddings.
    HDBSCAN is a density-based algorithm that can find clusters of varying shapes
    and handles noise (outliers) by assigning them a label of -1.

    Args:
        reduced_embeddings: A numpy array of reduced-dimension embeddings.
        min_cluster_size: The minimum size of clusters HDBSCAN should find.
                          Smaller values lead to more clusters, larger values to fewer.

    Returns:
        A numpy array of cluster labels, where -1 indicates noise samples.
    """
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric=distance_metric,
        cluster_selection_epsilon=0.0,  # Recommended by BERTopic for density-based selection
        prediction_data=False,  # We don't need prediction data for new points
        gen_min_span_tree=True,  # Set to True for better reproducibility with random_state
    )
    return clusterer.fit_predict(reduced_embeddings)


def label_documents(
    dataset: datasets.Dataset,
    cluster_labels: np.ndarray,
    cluster_id_column_name: str = "cluster_id",
) -> datasets.Dataset:
    """
    Adds cluster labels as a new column to the Hugging Face Dataset.
    The `datasets` library works efficiently with Apache Arrow tables, and `add_column`
    is the recommended way to modify datasets.

    Args:
        dataset: The Hugging Face Dataset object.
        cluster_labels: A numpy array of cluster labels corresponding to the documents.
        cluster_id_column_name: The name of the new column to be added for cluster IDs.

    Returns:
        The updated Hugging Face Dataset object with a new 'cluster_id' column.
    """
    # Hugging Face `Dataset` objects require the added column to be a list-like structure.
    # Ensure the length of labels matches the number of documents in the dataset.
    if len(cluster_labels) != len(dataset):
        raise ValueError(
            f"Mismatch between number of documents ({len(dataset)}) "
            f"and number of cluster labels ({len(cluster_labels)})."
        )
    # Convert numpy array to list for `add_column` method
    return dataset.add_column(cluster_id_column_name, cluster_labels.tolist())


def extract_keywords_ctfidf(
    dataset_with_labels: datasets.Dataset,
    text_column_name: str,
    cluster_id_column_name: str = "cluster_id",
    top_k_keywords: int = 10,
    stop_words: Union[str, List[str]] = "english",
) -> Dict[int, List[str]]:
    """
    Extracts top keywords for each cluster using a simplified Cluster-based TF-IDF (c-TF-IDF) approach.
    This implementation concatenates all documents within a cluster into a single "pseudo-document"
    and then applies a standard TF-IDF pipeline to these cluster pseudo-documents.

    Args:
        dataset_with_labels: Hugging Face Dataset with 'text_column_name' and 'cluster_id_column_name'.
        text_column_name: The name of the column containing the text documents.
        cluster_id_column_name: The name of the column containing cluster IDs.
        top_k_keywords: The number of top keywords to extract per cluster.
        stop_word_lang: Language specifier for the stopword list (passed to CountVectorizer).

    Returns:
        A dictionary where keys are cluster IDs (int) and values are lists of top_k_keywords (str).
    """
    # Convert the dataset to a pandas DataFrame for easier grouping and text concatenation.
    # This might be memory intensive for very large datasets, consider chunking if needed.
    df = dataset_with_labels.to_pandas()

    # Filter out noise documents (labeled -1) as they don't belong to any specific cluster.
    df_clustered = df[df[cluster_id_column_name] != -1]

    if df_clustered.empty:
        warnings.warn(
            "No valid clusters found (all documents are noise or dataset is empty after filtering).",
            RuntimeWarning
        )
        return

    # Group documents by cluster ID and concatenate their text into a single string for each cluster.
    # This creates a "pseudo-document" representing each cluster.
    cluster_docs = df_clustered.groupby(cluster_id_column_name)[text_column_name]\
        .apply(lambda x: " ".join(x)).to_dict()

    if not cluster_docs:
        warnings.warn(
            "No valid cluster pseudo-documents generated for keyword extraction.",
            RuntimeWarning
        )
        return

    # Prepare data for the vectorizer: a list of cluster pseudo-documents, ordered by cluster ID
    sorted_cluster_ids = sorted(cluster_docs.keys())
    cluster_texts = [cluster_docs[cid] for cid in sorted_cluster_ids]

    # CountVectorizer tokenizes and counts word occurrences.
    # TfidfTransformer then converts these counts into TF-IDF scores.
    # This treats each concatenated cluster text as a document for TF-IDF calculation.
    vectorizer_pipeline = Pipeline([
        # Filter out stop words and terms that appear in too few/many documents
        ('count', CountVectorizer(stop_words=stop_words, min_df=3, max_df=.9)),
        ('tfidf', TfidfTransformer())
    ])

    try:
        tfidf_matrix = vectorizer_pipeline.fit_transform(cluster_texts)
    except ValueError as e:
        warnings.warn(
            f"TF-IDF calculation error:\n{e}\n\nThis often means too few documents/words "
            "for vectorizer settings (e.g., min_df is too high).", RuntimeWarning
        )
        return

    # Get the feature names (words) learned by the CountVectorizer
    feature_names = vectorizer_pipeline['count'].get_feature_names_out()
    
    # Convert the sparse TF-IDF matrix to a dense array
    tfidf_array = tfidf_matrix.toarray()

    keywords_by_cluster = {}
    for i, cluster_id in enumerate(sorted_cluster_ids):
        # Get TF-IDF scores for the current cluster's pseudo-document
        cluster_tfidf_scores = tfidf_array[i]

        # Get indices of the top-k highest scores (descending order)
        top_indices = cluster_tfidf_scores.argsort()[-top_k_keywords:][::-1]

        # Retrieve the corresponding keywords
        top_keywords_with_scores = [(feature_names[idx], cluster_tfidf_scores[idx]) for idx in top_indices]
        
        # Filter out keywords with a score of 0 (can happen if top_k is larger than available unique terms)
        # and store only the word itself.
        top_keywords = [word for word, score in top_keywords_with_scores if score > 0]
        keywords_by_cluster[cluster_id] = top_keywords[:top_k_keywords]

    return keywords_by_cluster


def generate_llm_prompt(
    dataset_with_labels: datasets.Dataset,
    keywords_by_cluster: Dict[int, List[str]],
    text_column_name: str,
    jinja_env_dir: str,
    template_file_name: str,
    cluster_id_column_name: str = "cluster_id",
    sample_n_docs: int = 3,
    max_wordcount: int = 200,
    seed: int = 42
) -> str:
    """
    Generates an LLM prompt for naming each cluster in the provided dataset.

    Args:
        dataset_with_labels: Hugging Face Dataset with 'text_column_name' and 'cluster_id_column_name'.
        keywords_by_cluster: A dictionary mapping cluster IDs to lists of keywords.
        text_column_name: The name of the column containing the text documents.
        jinja_env_dir: Directory containing prompt template(s).
        template_file_name: Name of the Jinja file to use (must be contained in `jinja_env_dir`).
        cluster_id_column_name: The name of the column containing cluster IDs.
        sample_n_docs: The number of sample documents to include per cluster in the prompt.
        max_wordcount: Upper limit on the number of words to include in the example documents
    """
    # Convert to pandas DataFrame for easier querying and sampling
    df = dataset_with_labels.to_pandas()
    
    # Filter out clusters that have no extracted keywords
    relevant_clusters = [c for c in keywords_by_cluster.keys() if keywords_by_cluster[c]]
    df_clustered_relevant = df[df[cluster_id_column_name].isin(relevant_clusters)]

    if df_clustered_relevant.empty:
        warnings.warn(
            "No relevant clusters or documents to generate prompts for after filtering.",
            RuntimeWarning
        )
        return

    cluster_desc_list = []
    num_keywords = 1000
    # Iterate through clusters in a sorted order for consistent output
    for cluster_id in sorted(keywords_by_cluster.keys()):
        # Skip noise cluster (-1) and clusters without keywords
        if cluster_id == -1 or not keywords_by_cluster[cluster_id]:
            continue 

        cluster_keywords = keywords_by_cluster[cluster_id]
        if len(cluster_keywords) < num_keywords:
            num_keywords = len(cluster_keywords)
        
        # Get all documents belonging to the current cluster
        cluster_documents = df_clustered_relevant[
            df_clustered_relevant[cluster_id_column_name] == cluster_id
        ][text_column_name].tolist()
        
        # Randomly sample 'sample_n_docs' from the cluster documents.
        # Ensure reproducibility of samples.
        if len(cluster_documents) > sample_n_docs:
            np.random.seed(seed)
            sample_indices = np.random.choice(len(cluster_documents), sample_n_docs, replace=False)
            sample_docs = [cluster_documents[idx] for idx in sample_indices]
        else:
            sample_docs = cluster_documents  # If fewer than N docs, take all available

        # Format the prompt for the current cluster
        prompt_section = f"**Cluster ID: {cluster_id}**\n"
        prompt_section += f"Keywords: {', '.join(cluster_keywords)}\n\n"
        prompt_section += "Sample Documents:\n"
        for i, doc in enumerate(sample_docs):
            word_list = doc.split()
            num_words = len(word_list)
            truncated_word_list = word_list[:max_wordcount] + ["..."] if num_words > max_wordcount \
                else word_list  # Limit document length in prompt to avoid excessively long files
            truncated_doc = " ".join(truncated_word_list)
            prompt_section += f"  - Example document {i + 1}: {truncated_doc}\n"
        prompt_section += "\n"
        cluster_desc_list.append(prompt_section)

    jinja_env = Environment(loader=FileSystemLoader(jinja_env_dir))
    prompt_template = jinja_env.get_template(template_file_name)
    return prompt_template.render(
        cluster_desc_list=cluster_desc_list,
        num_keywords=num_keywords,
        num_examples=sample_n_docs
    )


def main():
    args = parse_arguments()
    warnings.simplefilter("ignore", category=(FutureWarning, UserWarning))
    logger = basic_logger_init("info" if args.verbose else "warn")

    if not Path(args.prompt_template_file).exists():
        raise FileNotFoundError(f"Invalid template path: {args.prompt_template_file}")
    
    logger.info("Loading Dataset: %s", args.dataset_id)
    if args.dataset_id.endswith(".npz"):  # precomputed embeddings
        if Path(args.dataset_id).exists():
            document_embeddings = np.load(args.dataset_id)
    else:
        try:
            try:
                ds = datasets.load_dataset(args.dataset_id)
            except ValueError:
                ds = datasets.load_from_disk(args.dataset_id)
            if isinstance(ds, datasets.DatasetDict):
                if "train" in ds:  # Datasets often come as a DatasetDict; we typically use the 'train' split.
                    initial_dataset = ds["train"]
                elif len(ds) > 0:  # If no 'train' split, use the first available split
                    initial_dataset = next(iter(ds.values()))
                    logger.warn(f"No 'train' split found. Using '{list(ds.keys())[0]}' split instead.")
                else:
                    raise ValueError(f"No data splits found in dataset '{args.dataset_id}'.")
            else:
                initial_dataset = ds
            if EXCLUDE_SOURCES:
                logger.info("Removing sources: %s", ", ".join(EXCLUDE_SOURCES))
                num_docs_init = initial_dataset.num_rows
                initial_dataset = initial_dataset.filter(
                    lambda instance: instance["source"] not in EXCLUDE_SOURCES, num_proc=8
                )
                logger.info("Done; num_rows %d -> %d", num_docs_init, initial_dataset.num_rows)
            if args.n_docs is not None:
                initial_dataset = initial_dataset.take(min(args.n_docs, len(initial_dataset)))
            # Retrieve all documents from the determined text column.
            documents = initial_dataset[args.text_column_name]
            num_documents = len(documents)
            if not documents:
                raise ValueError(f"The text column '{args.text_column_name}' is empty or could not be retrieved.")
            logger.info(
                "Using %d documents from the data field '%s'",
                num_documents, args.text_column_name
            )
        except Exception:
            logger.error("Failed to load text field '%s' from dataset '%s'", args.text_column_name, args.dataset_id)
            traceback.print_exc()
            exit(1)  # Exit if dataset loading fails, as pipeline cannot proceed
        output_path = make_version_subdir_path(Path(args.output_dir), make=True, stem="tm-run")
        with (output_path / "script_config.json").open("w") as f:
            json.dump(vars(args), f, indent=4)

        logger.info(
            "*** Embedding Documents ***\nModel: %s\n#documents %d",
            args.bert_model_id, num_documents
        )
        device = "cpu"
        if args.use_gpu:
            from torch import cuda

            if cuda.is_available():
                n_gpus = cuda.device_count()
                logger.info("GPUs: %d", n_gpus)
                device = [f"cuda:{i}" for i in range(n_gpus)] if n_gpus > 1 else "cuda:0"
            else:
                logger.warning("use_gpu=True but CUDA unavailable; using CPU")
        document_embeddings = embed_documents(
            documents=documents,
            bert_model_name=args.bert_model_id,
            batch_size=args.batch_size,
            pb=args.verbose,
            device=device
        )
        if args.save_embeddings:
            embeddings_dir = Path(_DATADIR_BASE) / "embeddings"
            embeddings_metadata = {"dataset": args.dataset_id, "model": args.bert_model_id}
            embeddings_output_path = save_embeddings(
                document_embeddings,
                embeddings_dir,
                ds["doc_id"],
                embeddings_metadata
            )
            logger.info("Document embeddings saved to disk @ %s", embeddings_output_path)
    logger.info("Embedding shape: %d x %d", *document_embeddings.shape)

    logger.info(
        "*** Dimensionality Reduction ***\nReducing dimensions to %d components with %d neighbors",
        args.umap_n_components, args.umap_n_neighbours
    )
    reduced_embeddings = reduce_dimensions(
        embeddings=document_embeddings,
        n_components=args.umap_n_components,
        n_neighbours=args.umap_n_neighbours,
        random_state=args.seed,
        verbose=args.verbose
    )
    logger.info("Reduced embedding shape: %d x %d", *reduced_embeddings.shape)

    logger.info(
        "*** Clustering (HDBSCAN) ***\nmin_cluster_size=%d",
        args.hdbscan_min_cluster_size
    )
    cluster_labels = perform_clustering(
        reduced_embeddings=reduced_embeddings,
        min_cluster_size=args.hdbscan_min_cluster_size,
    )
    n_clusters = len(np.unique(cluster_labels[cluster_labels != -1]))  # excluding noise (-1)
    n_labels_total, n_noise = len(cluster_labels), np.sum(cluster_labels == -1)
    noise_proportion = n_noise / n_labels_total
    logger.info(
        "Clusters found: %d (excluding noise); noise samples (label -1): %d / %d (pr=%.3f)",
        n_clusters, n_noise, n_labels_total, noise_proportion
    )

    logger.info("*** Labelling Documents ***")
    dataset_with_labels = label_documents(
        dataset=initial_dataset,
        cluster_labels=cluster_labels,
    )
    doc_ids_with_labels = dataset_with_labels.select_columns(["doc_id", "cluster_id"])
    doc_ids_with_labels.save_to_disk(output_path / "id2cluster")

    logger.info(
        "*** Keyword Extraction (c-TF-IDF) ***\nExtracting top %d keywords per cluster",
        args.ctfidf_top_k
    )
    stop_words = get_stop_words("french")
    keywords_by_cluster = extract_keywords_ctfidf(
        dataset_with_labels=dataset_with_labels,
        text_column_name=args.text_column_name,
        top_k_keywords=args.ctfidf_top_k,
        stop_words=stop_words
    )
    if keywords_by_cluster is None:
        logger.error("Keyword extraction unsuccessful; exiting")
        exit(1)
    num_clusters_with_keywords = len([k for k, v in keywords_by_cluster.items() if v])
    example_cluster_id = next(iter(keywords_by_cluster.keys()))
    with (output_path / "keywords.json").open("w") as f:
        json.dump(keywords_by_cluster, f, indent=4)
    if example_cluster_id is not None:
        logger.info(
            "Extracted keywords for %d clusters\nExample for Cluster %d: %s",
            num_clusters_with_keywords, example_cluster_id, keywords_by_cluster[example_cluster_id]
        )

    logger.info("*** Prompt Generation ***")
    jinja_env_dir, template_file_name = os.path.split(args.prompt_template_file)
    prompt = generate_llm_prompt(
        dataset_with_labels=dataset_with_labels,
        keywords_by_cluster=keywords_by_cluster,
        text_column_name=args.text_column_name,
        sample_n_docs=args.sample_n_docs,
        jinja_env_dir=jinja_env_dir,
        template_file_name=template_file_name,
        seed=args.seed
    )
    with (output_path / "generated-prompt.txt").open("w") as f:
        f.write(prompt)
    logger.info("Done; output @ %s\n%s", output_path, "=" * 70)


if __name__ == "__main__":
    main()

