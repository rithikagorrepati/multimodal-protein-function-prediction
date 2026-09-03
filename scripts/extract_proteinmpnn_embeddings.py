"""Extract per-residue ProteinMPNN representations from AlphaFold PDB files.

This script mirrors the structural-embedding step used in the original project:

AlphaFold PDB -> ProteinMPNN -> hidden structural representation

ProteinMPNN is an external dependency and is not included in this repository.
Clone the official ProteinMPNN repository separately and provide its path with
--proteinmpnn-dir.

Example
-------
python scripts/extract_proteinmpnn_embeddings.py \
    --pdb-dir data/structures \
    --proteinmpnn-dir /path/to/ProteinMPNN \
    --weights /path/to/ProteinMPNN/vanilla_model_weights/v_48_020.pt \
    --output-dir data/structure_embeddings
"""

from __future__ import annotations

import argparse
import copy
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch


ALPHAFOLD_ID_PATTERN = re.compile(
    r"^AF-([A-Za-z0-9]+)-F\d+-"
)


def infer_uniprot_id(
    pdb_path: Path,
) -> str:
    """Infer a UniProt accession from a standard AlphaFold filename."""

    match = ALPHAFOLD_ID_PATTERN.match(
        pdb_path.name
    )

    if match:
        return match.group(1)

    return pdb_path.stem


def load_proteinmpnn(
    proteinmpnn_dir: Path,
):
    """Import ProteinMPNN utilities from an external clone."""

    utils_path = (
        proteinmpnn_dir
        / "protein_mpnn_utils.py"
    )

    if not utils_path.exists():
        raise FileNotFoundError(
            "protein_mpnn_utils.py was not found in "
            f"{proteinmpnn_dir}. Clone the official "
            "ProteinMPNN repository and provide its "
            "root directory with --proteinmpnn-dir."
        )

    sys.path.insert(
        0,
        str(
            proteinmpnn_dir.resolve()
        ),
    )

    from protein_mpnn_utils import (
        ProteinMPNN,
        parse_PDB,
        tied_featurize,
    )

    return (
        ProteinMPNN,
        parse_PDB,
        tied_featurize,
    )


def build_model(
    ProteinMPNN,
    weights_path: Path,
    device: torch.device,
    ca_only: bool,
):
    """Load the pretrained ProteinMPNN model."""

    checkpoint = torch.load(
        weights_path,
        map_location=device,
    )

    hidden_dim = 128
    num_layers = 3

    model = ProteinMPNN(
        ca_only=ca_only,
        num_letters=21,
        node_features=hidden_dim,
        edge_features=hidden_dim,
        hidden_dim=hidden_dim,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        augment_eps=0.0,
        k_neighbors=checkpoint[
            "num_edges"
        ],
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.to(device)
    model.eval()

    return model


def extract_one_structure(
    pdb_path: Path,
    model,
    parse_PDB,
    tied_featurize,
    device: torch.device,
    ca_only: bool,
) -> np.ndarray:
    """Extract a ProteinMPNN encoder representation."""

    parsed = parse_PDB(
        str(pdb_path),
        ca_only=ca_only,
    )

    if not parsed:
        raise ValueError(
            f"ProteinMPNN could not parse "
            f"{pdb_path}"
        )

    protein = parsed[0]

    batch_clones = [
        copy.deepcopy(protein)
    ]

    captured: dict[
        str,
        torch.Tensor,
    ] = {}

    def hook(
        _module,
        _inputs,
        output,
    ):
        # Encoder layers return
        # (node_features, edge_features).
        if isinstance(
            output,
            (tuple, list),
        ):
            node_features = (
                output[0]
            )
        else:
            node_features = output

        captured[
            "node_features"
        ] = (
            node_features
            .detach()
            .cpu()
        )

    handle = (
        model
        .encoder_layers[0]
        .register_forward_hook(
            hook
        )
    )

    try:

        (
            X,
            S,
            mask,
            _lengths,
            chain_M,
            chain_encoding_all,
            _chain_list_list,
            _visible_list_list,
            _masked_list_list,
            _masked_chain_length_list_list,
            chain_M_pos,
            _omit_AA_mask,
            residue_idx,
            _dihedral_mask,
            _tied_pos_list_of_lists_list,
            _pssm_coef,
            _pssm_bias,
            _pssm_log_odds_all,
            _bias_by_res_all,
            _tied_beta,
        ) = tied_featurize(
            batch_clones,
            device,
            None,
        )

        random_noise = torch.randn(
            chain_M.shape,
            device=X.device,
        )

        with torch.no_grad():

            model(
                X,
                S,
                mask,
                chain_M
                * chain_M_pos,
                residue_idx,
                chain_encoding_all,
                random_noise,
            )

    finally:

        handle.remove()

    if (
        "node_features"
        not in captured
    ):
        raise RuntimeError(
            "No encoder representation "
            f"was captured for {pdb_path}"
        )

    representation = captured[
        "node_features"
    ]

    # Batch size is one.
    # Final shape:
    # [number_of_residues, 128]
    representation = (
        representation[0]
        .numpy()
        .astype(np.float32)
    )

    return representation


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Extract ProteinMPNN "
            "representations from "
            "AlphaFold PDB files."
        )
    )

    parser.add_argument(
        "--pdb-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing "
            "AlphaFold PDB files."
        ),
    )

    parser.add_argument(
        "--proteinmpnn-dir",
        type=Path,
        required=True,
        help=(
            "Root directory of an "
            "external ProteinMPNN clone."
        ),
    )

    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help=(
            "ProteinMPNN checkpoint, "
            "for example "
            "vanilla_model_weights/"
            "v_48_020.pt."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/"
            "structure_embeddings"
        ),
    )

    parser.add_argument(
        "--ca-only",
        action="store_true",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    random.seed(
        args.seed
    )

    np.random.seed(
        args.seed
    )

    torch.manual_seed(
        args.seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            args.seed
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if not args.pdb_dir.exists():

        raise FileNotFoundError(
            "PDB directory does not exist: "
            f"{args.pdb_dir}"
        )

    if not args.weights.exists():

        raise FileNotFoundError(
            "ProteinMPNN weights "
            "do not exist: "
            f"{args.weights}"
        )

    pdb_files = sorted(
        args.pdb_dir.glob(
            "*.pdb"
        )
    )

    if not pdb_files:

        raise FileNotFoundError(
            "No .pdb files found in "
            f"{args.pdb_dir}"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        ProteinMPNN,
        parse_PDB,
        tied_featurize,
    ) = load_proteinmpnn(
        args.proteinmpnn_dir
    )

    model = build_model(
        ProteinMPNN,
        args.weights,
        device,
        args.ca_only,
    )

    print(
        f"Device: {device}"
    )

    print(
        f"PDB files: "
        f"{len(pdb_files)}"
    )

    successful = 0
    failed = []

    for index, pdb_path in enumerate(
        pdb_files,
        start=1,
    ):

        uniprot_id = infer_uniprot_id(
            pdb_path
        )

        output_path = (
            args.output_dir
            / (
                f"{uniprot_id}"
                "_embedding.npy"
            )
        )

        try:

            representation = (
                extract_one_structure(
                    pdb_path=pdb_path,
                    model=model,
                    parse_PDB=parse_PDB,
                    tied_featurize=tied_featurize,
                    device=device,
                    ca_only=args.ca_only,
                )
            )

            np.save(
                output_path,
                representation,
            )

            successful += 1

            print(
                f"[{index}/"
                f"{len(pdb_files)}] "
                f"{uniprot_id}: "
                f"{representation.shape}"
            )

        except Exception as exc:

            failed.append(
                (
                    pdb_path.name,
                    str(exc),
                )
            )

            print(
                f"[{index}/"
                f"{len(pdb_files)}] "
                f"FAILED "
                f"{pdb_path.name}: "
                f"{exc}"
            )

    print(
        "\nExtraction complete."
    )

    print(
        f"Successful: {successful}"
    )

    print(
        f"Failed: {len(failed)}"
    )

    if failed:

        failure_log = (
            args.output_dir
            / "failed_structures.tsv"
        )

        with failure_log.open(
            "w",
            encoding="utf-8",
        ) as handle:

            handle.write(
                "pdb_file\terror\n"
            )

            for (
                pdb_file,
                error,
            ) in failed:

                handle.write(
                    f"{pdb_file}\t"
                    f"{error}\n"
                )

        print(
            f"Failure log: "
            f"{failure_log}"
        )


if __name__ == "__main__":
    main()
