"""Generate BioBERT embeddings for UniProt protein annotations.

Workflow
--------
UniProt functional annotation
-> BioBERT
-> sentence-level embedding
-> annotation classifier

The source metadata can contain multiple rows for the same UniProt accession.
The cleaned workflow keeps one annotation per protein so all modalities can be
aligned using the shared UniProt-level split manifest.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


DEFAULT_MODEL = "dmis-lab/biobert-base-cased-v1.1"


def load_annotation_metadata(
    metadata_path: Path,
) -> pd.DataFrame:
    """Load and clean protein annotation metadata."""

    metadata = pd.read_csv(
        metadata_path,
        sep="\t",
    )

    required_columns = {
        "UniProtID",
        "Descriptor",
        "EC",
    }

    missing_columns = (
        required_columns
        .difference(metadata.columns)
    )

    if missing_columns:
        raise ValueError(
            "Metadata is missing required columns: "
            + ", ".join(
                sorted(
                    missing_columns
                )
            )
        )

    metadata = (
        metadata
        .dropna(
            subset=[
                "UniProtID",
                "Descriptor",
                "EC",
            ]
        )
        .copy()
    )

    metadata["UniProtID"] = (
        metadata["UniProtID"]
        .astype(str)
        .str.strip()
    )

    metadata["Descriptor"] = (
        metadata["Descriptor"]
        .astype(str)
        .str.strip()
    )

    metadata["EC"] = (
        metadata["EC"]
        .astype(str)
        .str.strip()
    )

    metadata = metadata[
        metadata["Descriptor"]
        .str.len()
        > 0
    ].copy()

    # A UniProt accession should map to one EC label in this dataset.
    ec_counts = (
        metadata
        .groupby(
            "UniProtID"
        )["EC"]
        .nunique()
    )

    conflicting_ids = (
        ec_counts[
            ec_counts > 1
        ]
        .index
        .tolist()
    )

    if conflicting_ids:
        raise ValueError(
            "Some UniProt IDs have conflicting EC labels: "
            + ", ".join(
                conflicting_ids[:10]
            )
        )

    original_rows = len(
        metadata
    )

    # Some accessions appear multiple times because of gene aliases.
    # If duplicate descriptions differ, retain the longest available
    # annotation for that protein.
    metadata[
        "_descriptor_length"
    ] = (
        metadata[
            "Descriptor"
        ]
        .str.len()
    )

    metadata = (
        metadata
        .sort_values(
            [
                "UniProtID",
                "_descriptor_length",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .drop_duplicates(
            subset=[
                "UniProtID"
            ],
            keep="first",
        )
        .drop(
            columns=[
                "_descriptor_length"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    duplicate_rows_removed = (
        original_rows
        - len(metadata)
    )

    print(
        f"Usable annotation rows before "
        f"UniProt deduplication: "
        f"{original_rows}"
    )

    print(
        f"Unique proteins retained: "
        f"{len(metadata)}"
    )

    print(
        f"Duplicate rows removed: "
        f"{duplicate_rows_removed}"
    )

    print(
        f"Distinct EC classes: "
        f"{metadata['EC'].nunique()}"
    )

    return metadata


def mean_pool_last_hidden_state(
    hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Average final-layer vectors across non-padding tokens."""

    mask = (
        attention_mask
        .unsqueeze(-1)
        .to(
            hidden_state.dtype
        )
    )

    summed = (
        hidden_state
        * mask
    ).sum(
        dim=1
    )

    counts = (
        mask
        .sum(dim=1)
        .clamp(
            min=1.0
        )
    )

    return (
        summed
        / counts
    )


def generate_embeddings(
    descriptions: list[str],
    model_name: str,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Generate one BioBERT embedding per protein annotation."""

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            model_name
        )
    )

    model = (
        AutoModel
        .from_pretrained(
            model_name
        )
        .to(device)
    )

    model.eval()

    embedding_batches = []

    with torch.no_grad():

        for start in range(
            0,
            len(descriptions),
            batch_size,
        ):

            batch = descriptions[
                start:
                start + batch_size
            ]

            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )

            encoded = {
                key:
                value.to(
                    device
                )
                for key, value
                in encoded.items()
            }

            outputs = model(
                **encoded
            )

            pooled = (
                mean_pool_last_hidden_state(
                    outputs.last_hidden_state,
                    encoded[
                        "attention_mask"
                    ],
                )
            )

            embedding_batches.append(
                pooled
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.float32
                )
            )

            processed = min(
                start
                + batch_size,
                len(
                    descriptions
                ),
            )

            print(
                f"Processed "
                f"{processed}/"
                f"{len(descriptions)} "
                f"annotations"
            )

    if not embedding_batches:
        raise ValueError(
            "No BioBERT embeddings "
            "were generated."
        )

    embeddings = np.concatenate(
        embedding_batches,
        axis=0,
    )

    return embeddings


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Generate BioBERT embeddings "
            "from UniProt protein annotations."
        )
    )

    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help=(
            "Project metadata TSV containing "
            "UniProtID, Descriptor, and EC."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/annotation"
        ),
        help=(
            "Directory for annotation "
            "metadata and embeddings."
        ),
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "Hugging Face BioBERT "
            "checkpoint."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
        default="auto",
    )

    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError(
            "--batch-size must "
            "be at least 1."
        )

    if args.device == "auto":

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    elif args.device == "cuda":

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but "
                "is not available."
            )

        device = torch.device(
            "cuda"
        )

    else:

        device = torch.device(
            "cpu"
        )

    metadata = (
        load_annotation_metadata(
            args.metadata
        )
    )

    print(
        f"BioBERT checkpoint: "
        f"{args.model}"
    )

    print(
        f"Device: "
        f"{device}"
    )

    embeddings = (
        generate_embeddings(
            descriptions=metadata[
                "Descriptor"
            ].tolist(),
            model_name=args.model,
            batch_size=args.batch_size,
            device=device,
        )
    )

    if (
        len(metadata)
        != len(embeddings)
    ):
        raise RuntimeError(
            "Annotation metadata and "
            "embedding counts differ."
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Keep the row position explicit so the connection between
    # annotation_metadata.csv and biobert_embeddings.npy is clear.
    metadata_output = (
        metadata[
            [
                "UniProtID",
                "Descriptor",
                "EC",
            ]
        ]
        .copy()
    )

    metadata_output.insert(
        0,
        "EmbeddingRow",
        np.arange(
            len(
                metadata_output
            )
        ),
    )

    metadata_path = (
        args.output_dir
        / "annotation_metadata.csv"
    )

    embeddings_path = (
        args.output_dir
        / "biobert_embeddings.npy"
    )

    metadata_output.to_csv(
        metadata_path,
        index=False,
    )

    np.save(
        embeddings_path,
        embeddings,
    )

    print(
        "\nEmbedding generation complete."
    )

    print(
        f"Embedding matrix shape: "
        f"{embeddings.shape}"
    )

    print(
        f"Metadata saved to: "
        f"{metadata_path}"
    )

    print(
        f"Embeddings saved to: "
        f"{embeddings_path}"
    )


if __name__ == "__main__":
    main()
