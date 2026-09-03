"""Generate BioBERT sentence embeddings for protein annotations.

The original project used the ``biobert_embedding`` package, whose
``sentence_vector`` method averages the final hidden-layer token vectors.
This script reproduces that sentence-level representation with Hugging Face
Transformers and BioBERT v1.1.

Example
-------
python scripts/extract_biobert_embeddings.py \
    --metadata Parsed_EC_Descriptor.tsv \
    --output-dir data/annotation
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


DEFAULT_MODEL = "dmis-lab/biobert-base-cased-v1.1"


def mean_pool_last_hidden_state(
    hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Average final-layer token vectors over non-padding tokens."""

    mask = attention_mask.unsqueeze(-1).to(hidden_state.dtype)
    summed = (hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)

    return summed / counts


def generate_embeddings(
    descriptions: list[str],
    model_name: str,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Generate one BioBERT vector per protein description."""

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    batches = []

    with torch.no_grad():

        for start in range(0, len(descriptions), batch_size):

            batch = descriptions[start : start + batch_size]

            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )

            encoded = {
                key: value.to(device)
                for key, value in encoded.items()
            }

            outputs = model(**encoded)

            pooled = mean_pool_last_hidden_state(
                outputs.last_hidden_state,
                encoded["attention_mask"],
            )

            batches.append(
                pooled
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            processed = min(
                start + batch_size,
                len(descriptions),
            )

            print(
                f"Processed {processed}/"
                f"{len(descriptions)} annotations"
            )

    return np.concatenate(
        batches,
        axis=0,
    )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Generate BioBERT embeddings "
            "for protein annotations."
        )
    )

    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help=(
            "TSV containing UniProtID, "
            "Descriptor, and EC columns."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/annotation"),
        help=(
            "Directory for generated "
            "metadata and embeddings."
        ),
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Hugging Face BioBERT checkpoint.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )

    args = parser.parse_args()

    if args.device == "auto":

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    else:

        device = torch.device(
            args.device
        )

    metadata = pd.read_csv(
        args.metadata,
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
                sorted(missing_columns)
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
    )

    metadata["EC"] = (
        metadata["EC"]
        .astype(str)
        .str.strip()
    )

    print(
        f"Annotation records: "
        f"{len(metadata)}"
    )

    print(
        f"Distinct EC classes: "
        f"{metadata['EC'].nunique()}"
    )

    print(
        f"Device: {device}"
    )

    embeddings = generate_embeddings(
        descriptions=metadata[
            "Descriptor"
        ].tolist(),
        model_name=args.model,
        batch_size=args.batch_size,
        device=device,
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_output = (
        args.output_dir
        / "annotation_metadata.csv"
    )

    embeddings_output = (
        args.output_dir
        / "biobert_embeddings.npy"
    )

    metadata[
        [
            "UniProtID",
            "Descriptor",
            "EC",
        ]
    ].to_csv(
        metadata_output,
        index=False,
    )

    np.save(
        embeddings_output,
        embeddings,
    )

    print(
        f"Embedding matrix shape: "
        f"{embeddings.shape}"
    )

    print(
        f"Saved metadata to: "
        f"{metadata_output}"
    )

    print(
        f"Saved embeddings to: "
        f"{embeddings_output}"
    )


if __name__ == "__main__":
    main()
