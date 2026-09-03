"""Create a shared train/validation/test split for all model branches.

The split is generated once at the protein (UniProt ID) level and can then be
reused by the sequence, annotation, structure, and multimodal pipelines.

Example
-------
python scripts/create_split_manifest.py \
    --metadata Parsed_EC_Descriptor.tsv \
    --output data/split_manifest.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Create a shared protein-level "
            "train/validation/test split."
        )
    )

    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="Master metadata TSV.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/split_manifest.csv"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    metadata = pd.read_csv(
        args.metadata,
        sep="\t",
    )

    required_columns = {
        "UniProtID",
        "EC",
    }

    missing = required_columns.difference(
        metadata.columns
    )

    if missing:
        raise ValueError(
            "Metadata is missing required columns: "
            + ", ".join(sorted(missing))
        )

    proteins = (
        metadata[
            [
                "UniProtID",
                "EC",
            ]
        ]
        .dropna()
        .copy()
    )

    proteins["UniProtID"] = (
        proteins["UniProtID"]
        .astype(str)
        .str.strip()
    )

    proteins["EC"] = (
        proteins["EC"]
        .astype(str)
        .str.strip()
    )

    # One split assignment per protein.
    proteins = (
        proteins
        .drop_duplicates(
            subset=["UniProtID"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    train_val, test = train_test_split(
        proteins,
        test_size=0.20,
        random_state=args.seed,
        shuffle=True,
    )

    train, validation = train_test_split(
        train_val,
        test_size=0.125,
        random_state=args.seed,
        shuffle=True,
    )

    train = train.assign(
        split="train"
    )

    validation = validation.assign(
        split="validation"
    )

    test = test.assign(
        split="test"
    )

    manifest = pd.concat(
        [
            train,
            validation,
            test,
        ],
        ignore_index=True,
    )

    manifest = manifest.sort_values(
        "UniProtID"
    ).reset_index(
        drop=True
    )

    if (
        manifest["UniProtID"]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "A UniProt ID was assigned "
            "to more than one split."
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest.to_csv(
        args.output,
        index=False,
    )

    counts = (
        manifest["split"]
        .value_counts()
    )

    print(
        f"Unique proteins: "
        f"{len(manifest)}"
    )

    print(
        f"Train: "
        f"{counts.get('train', 0)}"
    )

    print(
        f"Validation: "
        f"{counts.get('validation', 0)}"
    )

    print(
        f"Test: "
        f"{counts.get('test', 0)}"
    )

    print(
        f"Distinct EC classes: "
        f"{manifest['EC'].nunique()}"
    )

    print(
        f"Saved split manifest to: "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()
