"""Build the protein sequence classification dataset.

This script joins UniProt FASTA sequences with Enzyme Commission (EC)
labels from the project metadata.

Example
-------
python scripts/build_sequence_dataset.py \
    --metadata Parsed_EC_Descriptor.tsv \
    --sequence-dir data/sequences \
    --output data/sequence_dataset.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def read_fasta(path: Path) -> str:
    """Read a single-record FASTA file and return its amino-acid sequence."""

    lines = path.read_text(encoding="utf-8").splitlines()

    sequence = "".join(
        line.strip()
        for line in lines
        if line.strip() and not line.startswith(">")
    )

    return sequence


def find_fasta(sequence_dir: Path, uniprot_id: str) -> Path | None:
    """Find a FASTA file for a UniProt accession."""

    for extension in (".fasta", ".fa", ".faa"):
        path = sequence_dir / f"{uniprot_id}{extension}"

        if path.exists():
            return path

    return None


def build_sequence_dataset(
    metadata_path: Path,
    sequence_dir: Path,
) -> pd.DataFrame:
    """Join downloaded protein sequences with EC labels."""

    metadata = pd.read_csv(metadata_path, sep="\t")

    required_columns = {"UniProtID", "EC"}

    missing_columns = required_columns.difference(metadata.columns)

    if missing_columns:
        raise ValueError(
            f"Metadata is missing required columns: "
            f"{', '.join(sorted(missing_columns))}"
        )

    metadata = metadata.dropna(subset=["UniProtID", "EC"]).copy()

    metadata["UniProtID"] = metadata["UniProtID"].astype(str).str.strip()
    metadata["EC"] = metadata["EC"].astype(str).str.strip()

    # Some UniProt accessions occur more than once in the source metadata.
    # Each accession is represented once in the sequence-classification table.
    metadata = metadata.drop_duplicates(subset=["UniProtID"], keep="first")

    records = []
    missing_sequences = []

    for row in metadata.itertuples(index=False):
        uniprot_id = row.UniProtID

        fasta_path = find_fasta(sequence_dir, uniprot_id)

        if fasta_path is None:
            missing_sequences.append(uniprot_id)
            continue

        sequence = read_fasta(fasta_path)

        if not sequence:
            missing_sequences.append(uniprot_id)
            continue

        records.append(
            {
                "UniProtID": uniprot_id,
                "Sequence": sequence,
                "EC": row.EC,
            }
        )

    dataset = pd.DataFrame(
        records,
        columns=["UniProtID", "Sequence", "EC"],
    )

    print(f"Metadata proteins with EC labels: {len(metadata)}")
    print(f"Usable sequence records: {len(dataset)}")
    print(f"Missing or empty FASTA files: {len(missing_sequences)}")
    print(f"Distinct EC classes: {dataset['EC'].nunique()}")

    if missing_sequences:
        print("\nFirst missing UniProt IDs:")
        for uniprot_id in missing_sequences[:10]:
            print(f"  {uniprot_id}")

    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Join UniProt FASTA sequences with EC labels."
    )

    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="Project metadata TSV containing UniProtID and EC columns.",
    )

    parser.add_argument(
        "--sequence-dir",
        type=Path,
        required=True,
        help="Directory containing UniProt FASTA files.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/sequence_dataset.csv"),
        help="Output CSV path.",
    )

    args = parser.parse_args()

    dataset = build_sequence_dataset(
        metadata_path=args.metadata,
        sequence_dir=args.sequence_dir,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output, index=False)

    print(f"\nSaved dataset to: {args.output}")


if __name__ == "__main__":
    main()
