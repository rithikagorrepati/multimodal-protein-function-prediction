"""Extract per-residue ProteinMPNN representations from AlphaFold PDB files.

ProteinMPNN is an external dependency. Clone the official ProteinMPNN
repository separately and provide its root directory with --proteinmpnn-dir.

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

    return (
        match.group(1)
        if match
        else pdb_path.stem
    )


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
            f"protein_mpnn_utils.py was not found in "
            f"{proteinmpnn_dir}. Clone the official "
            "ProteinMPNN repository and pass its root "
            "directory with --proteinmpnn-dir."
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
    """Load a pretrained ProteinMPNN model."""

    checkpoint = torch.load(
        weights_path,
        map_location=device,
    )

    model = ProteinMPNN(
        ca_only=ca_only,
        num_letters=21,
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
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

    model.to(
        device
    )

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
    """Extract the first ProteinMPNN encoder-layer representation."""

    parsed = parse_PDB(
        str(
            pdb_path
        ),
        ca_only=ca_only,
    )

    if not parsed:
        raise ValueError(
            f"ProteinMPNN could not parse "
            f"{pdb_path}"
        )

    batch_clones = [
        copy.deepcopy(
            parsed[0]
        )
    ]

    captured: dict[
        str,
        torch.Tensor,
    ] = {}

    def capture_encoder_output(
        _module,
        _inputs,
        output,
    ) -> None:

        node_features = (
            output[0]
            if isinstance(
                output,
                (tuple, list),
            )
            else output
        )

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
            capture_encoder_output
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
            ca_only=ca_only,
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
            "No ProteinMPNN encoder "
            "representation was captured "
            f"for {pdb_path}"
        )

    representation = (
        captured[
            "node_features"
        ][0]
        .numpy()
        .astype(
            np.float32
        )
    )

    if representation.ndim != 2:

        raise RuntimeError(
            f"Unexpected embedding shape "
            f"for {pdb_path}: "
            f"{representation.shape}"
        )

    return representation


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Extract ProteinMPNN representations "
            "from AlphaFold PDB files."
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
            "Root directory of an external "
            "ProteinMPNN clone."
        ),
    )

    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help=(
            "ProteinMPNN checkpoint, for example "
            "vanilla_model_weights/v_48_020.pt."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/structure_embeddings"
        ),
    )

    parser.add_argument(
        "--ca-only",
        action="store_true",
        help=(
            "Use ProteinMPNN's CA-only mode. "
            "When enabled, provide a compatible "
            "CA-only checkpoint."
        ),
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
            "ProteinMPNN weights do not exist: "
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
        ProteinMPNN=ProteinMPNN,
        weights_path=args.weights,
        device=device,
        ca_only=args.ca_only,
    )

    print(
        f"Device: {device}"
    )

    print(
        f"PDB files: "
        f"{len(pdb_files)}"
    )

    successful = 0

    failed: list[
        tuple[
            str,
            str,
        ]
    ] = []

    for (
        index,
        pdb_path,
    ) in enumerate(
        pdb_files,
        start=1,
    ):

        uniprot_id = (
            infer_uniprot_id(
                pdb_path
            )
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
                    tied_featurize=(
                        tied_featurize
                    ),
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
        f"Successful: "
        f"{successful}"
    )

    print(
        f"Failed: "
        f"{len(failed)}"
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

                clean_error = (
                    error
                    .replace(
                        "\t",
                        " ",
                    )
                    .replace(
                        "\n",
                        " ",
                    )
                )

                handle.write(
                    f"{pdb_file}\t"
                    f"{clean_error}\n"
                )

        print(
            f"Failure log: "
            f"{failure_log}"
        )


if __name__ == "__main__":
    main()
