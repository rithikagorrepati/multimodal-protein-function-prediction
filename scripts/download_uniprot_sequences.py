"""Download protein FASTA sequences from UniProt.

The input metadata file must contain a `UniProtID` column. Sequences are
downloaded from the UniProt REST API and saved as individual FASTA files.

Example
-------
python scripts/download_uniprot_sequences.py \
    --metadata Parsed_EC_Descriptor.tsv \
    --output-dir data/sequences
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests


UNIPROT_FASTA_URL = "https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"


def load_uniprot_ids(metadata_path: Path) -> list[str]:
    """Read unique UniProt IDs from a CSV or TSV metadata file."""

    separator = "\t" if metadata_path.suffix.lower() in {".tsv", ".txt"} else ","
    metadata = pd.read_csv(metadata_path, sep=separator)

    if "UniProtID" not in metadata.columns:
        raise ValueError("Metadata file must contain a 'UniProtID' column.")

    ids = (
        metadata["UniProtID"]
        .dropna()
        .astype(str)
        .str.strip()
        .drop_duplicates()
        .tolist()
    )

    return ids


def download_sequence(
    uniprot_id: str,
    output_dir: Path,
    session: requests.Session,
    overwrite: bool = False,
) -> bool:
    """Download one UniProt FASTA record."""

    output_path = output_dir / f"{uniprot_id}.fasta"

    if output_path.exists() and not overwrite:
        return True

    url = UNIPROT_FASTA_URL.format(uniprot_id=uniprot_id)

    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"Failed: {uniprot_id} ({exc})")
        return False

    output_path.write_text(response.text, encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download protein FASTA sequences from UniProt."
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="CSV or TSV file containing a UniProtID column.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/sequences"),
        help="Directory for downloaded FASTA files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload FASTA files that already exist.",
    )

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    uniprot_ids = load_uniprot_ids(args.metadata)

    print(f"Found {len(uniprot_ids)} unique UniProt IDs.")

    successful = 0
    failed = 0

    with requests.Session() as session:
        for index, uniprot_id in enumerate(uniprot_ids, start=1):
            success = download_sequence(
                uniprot_id,
                args.output_dir,
                session,
                overwrite=args.overwrite,
            )

            if success:
                successful += 1
            else:
                failed += 1

            if index % 100 == 0 or index == len(uniprot_ids):
                print(
                    f"Processed {index}/{len(uniprot_ids)} "
                    f"(successful: {successful}, failed: {failed})"
                )

            time.sleep(0.05)

    print("\nDownload complete.")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
