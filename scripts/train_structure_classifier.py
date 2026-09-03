"""Train a Transformer classifier on ProteinMPNN structural embeddings.

Workflow
--------
AlphaFold-predicted PDB structure
-> ProteinMPNN per-residue representation
-> Transformer encoder
-> Enzyme Commission (EC) prediction

All model branches use a shared UniProt-level split manifest so proteins are
assigned consistently across sequence, annotation, structure, and multimodal
workflows.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


class StructureDataset(Dataset):
    """Variable-length ProteinMPNN embeddings paired with EC labels."""

    def __init__(
        self,
        ids: list[str],
        embeddings: list[np.ndarray],
        labels: np.ndarray,
        max_residues: int | None = None,
    ) -> None:

        self.ids = ids
        self.embeddings = embeddings
        self.labels = labels
        self.max_residues = max_residues

    def __len__(self) -> int:

        return len(
            self.ids
        )

    def __getitem__(
        self,
        index: int,
    ):

        embedding = self.embeddings[
            index
        ]

        if self.max_residues is not None:

            embedding = embedding[
                : self.max_residues
            ]

        return (
            self.ids[index],
            torch.tensor(
                embedding,
                dtype=torch.float32,
            ),
            torch.tensor(
                self.labels[index],
                dtype=torch.long,
            ),
        )


def collate_structures(
    batch,
):
    """Pad variable-length residue embeddings within a batch."""

    ids, embeddings, labels = zip(
        *batch
    )

    lengths = torch.tensor(
        [
            embedding.shape[0]
            for embedding
            in embeddings
        ],
        dtype=torch.long,
    )

    padded = pad_sequence(
        embeddings,
        batch_first=True,
        padding_value=0.0,
    )

    max_length = padded.shape[1]

    positions = torch.arange(
        max_length
    ).unsqueeze(
        0
    )

    padding_mask = (
        positions
        >= lengths.unsqueeze(
            1
        )
    )

    return (
        list(ids),
        padded,
        padding_mask,
        torch.stack(
            labels
        ),
    )


class StructureTransformer(nn.Module):
    """Transformer classifier over ProteinMPNN residue representations."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:

        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.classifier = nn.Linear(
            input_dim,
            num_classes,
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:

        encoded = self.encoder(
            embeddings,
            src_key_padding_mask=padding_mask,
        )

        valid_mask = (
            ~padding_mask
        ).unsqueeze(
            -1
        ).to(
            encoded.dtype
        )

        pooled = (
            (
                encoded
                * valid_mask
            )
            .sum(
                dim=1
            )
            / valid_mask
            .sum(
                dim=1
            )
            .clamp(
                min=1.0
            )
        )

        logits = self.classifier(
            pooled
        )

        return logits


def set_seeds(
    seed: int,
) -> None:
    """Set random seeds for reproducibility."""

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


def load_structure_data(
    metadata_path: Path,
    embedding_dir: Path,
    split_manifest_path: Path,
):
    """Match ProteinMPNN embeddings to metadata and the shared split."""

    metadata = pd.read_csv(
        metadata_path,
        sep="\t",
    )

    required_metadata_columns = {
        "UniProtID",
        "EC",
    }

    missing_metadata_columns = (
        required_metadata_columns
        .difference(
            metadata.columns
        )
    )

    if missing_metadata_columns:

        raise ValueError(
            "Metadata is missing required columns: "
            + ", ".join(
                sorted(
                    missing_metadata_columns
                )
            )
        )

    metadata = (
        metadata
        .dropna(
            subset=[
                "UniProtID",
                "EC",
            ]
        )
        .copy()
    )

    metadata["UniProtID"] = (
        metadata[
            "UniProtID"
        ]
        .astype(str)
        .str.strip()
    )

    metadata["EC"] = (
        metadata[
            "EC"
        ]
        .astype(str)
        .str.strip()
    )

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

    metadata = (
        metadata
        .drop_duplicates(
            subset=[
                "UniProtID"
            ],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    ec_by_id = (
        metadata
        .set_index(
            "UniProtID"
        )["EC"]
        .to_dict()
    )

    manifest = pd.read_csv(
        split_manifest_path
    )

    required_manifest_columns = {
        "UniProtID",
        "EC",
        "split",
    }

    missing_manifest_columns = (
        required_manifest_columns
        .difference(
            manifest.columns
        )
    )

    if missing_manifest_columns:

        raise ValueError(
            "Split manifest is missing required columns: "
            + ", ".join(
                sorted(
                    missing_manifest_columns
                )
            )
        )

    manifest["UniProtID"] = (
        manifest[
            "UniProtID"
        ]
        .astype(str)
        .str.strip()
    )

    manifest["EC"] = (
        manifest[
            "EC"
        ]
        .astype(str)
        .str.strip()
    )

    manifest["split"] = (
        manifest[
            "split"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    if (
        manifest[
            "UniProtID"
        ]
        .duplicated()
        .any()
    ):

        raise ValueError(
            "Split manifest contains duplicate UniProt IDs."
        )

    valid_splits = {
        "train",
        "validation",
        "test",
    }

    unexpected_splits = (
        set(
            manifest[
                "split"
            ]
        )
        - valid_splits
    )

    if unexpected_splits:

        raise ValueError(
            "Unexpected split labels: "
            + ", ".join(
                sorted(
                    unexpected_splits
                )
            )
        )

    manifest_by_id = (
        manifest
        .set_index(
            "UniProtID"
        )
    )

    ids = []
    embeddings = []
    ec_labels = []
    split_labels = []

    skipped_no_metadata = 0
    skipped_no_manifest = 0
    invalid_embeddings = []

    embedding_files = sorted(
        embedding_dir.glob(
            "*_embedding.npy"
        )
    )

    if not embedding_files:

        raise FileNotFoundError(
            "No *_embedding.npy files were found in "
            f"{embedding_dir}"
        )

    for path in embedding_files:

        uniprot_id = (
            path.name
            .removesuffix(
                "_embedding.npy"
            )
        )

        if uniprot_id not in ec_by_id:

            skipped_no_metadata += 1
            continue

        if (
            uniprot_id
            not in manifest_by_id.index
        ):

            skipped_no_manifest += 1
            continue

        project_ec = ec_by_id[
            uniprot_id
        ]

        manifest_ec = str(
            manifest_by_id.loc[
                uniprot_id,
                "EC",
            ]
        )

        if (
            project_ec
            != manifest_ec
        ):

            raise ValueError(
                f"EC label mismatch for "
                f"{uniprot_id}: "
                f"{project_ec} vs "
                f"{manifest_ec}"
            )

        array = np.load(
            path
        )

        if (
            array.ndim != 2
            or array.shape[0] == 0
            or array.shape[1] == 0
        ):

            invalid_embeddings.append(
                path.name
            )

            continue

        ids.append(
            uniprot_id
        )

        embeddings.append(
            array.astype(
                np.float32
            )
        )

        ec_labels.append(
            project_ec
        )

        split_labels.append(
            str(
                manifest_by_id.loc[
                    uniprot_id,
                    "split",
                ]
            )
        )

    if not ids:

        raise ValueError(
            "No valid ProteinMPNN embeddings "
            "could be matched to the project metadata "
            "and shared split manifest."
        )

    feature_dimensions = {
        array.shape[1]
        for array
        in embeddings
    }

    if len(
        feature_dimensions
    ) != 1:

        raise ValueError(
            "ProteinMPNN embeddings do not all "
            "have the same feature dimension."
        )

    combined = pd.DataFrame(
        {
            "UniProtID":
            ids,

            "EC":
            ec_labels,

            "split":
            split_labels,

            "embedding_index":
            np.arange(
                len(ids)
            ),
        }
    )

    print(
        f"ProteinMPNN files found: "
        f"{len(embedding_files)}"
    )

    print(
        f"Matched structures: "
        f"{len(combined)}"
    )

    print(
        f"Skipped without metadata: "
        f"{skipped_no_metadata}"
    )

    print(
        f"Skipped without split assignment: "
        f"{skipped_no_manifest}"
    )

    print(
        f"Invalid embeddings: "
        f"{len(invalid_embeddings)}"
    )

    return (
        combined,
        embeddings,
        manifest,
    )


def build_dataset(
    metadata: pd.DataFrame,
    embeddings: list[np.ndarray],
    split_name: str,
    labels: np.ndarray,
    max_residues: int | None,
) -> StructureDataset:
    """Create one structure dataset from the shared split."""

    mask = (
        metadata[
            "split"
        ]
        == split_name
    )

    indices = np.flatnonzero(
        mask.to_numpy()
    )

    if len(
        indices
    ) == 0:

        raise ValueError(
            f"No structure records found "
            f"for split '{split_name}'."
        )

    selected_ids = [
        metadata.iloc[
            index
        ]["UniProtID"]
        for index
        in indices
    ]

    selected_embeddings = [
        embeddings[
            index
        ]
        for index
        in indices
    ]

    selected_labels = labels[
        indices
    ]

    return StructureDataset(
        ids=selected_ids,
        embeddings=selected_embeddings,
        labels=selected_labels,
        max_residues=max_residues,
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
):
    """Evaluate the structure classifier."""

    model.eval()

    all_ids = []
    all_logits = []
    all_labels = []

    with torch.no_grad():

        for (
            ids,
            embeddings,
            padding_mask,
            labels,
        ) in loader:

            embeddings = embeddings.to(
                device
            )

            padding_mask = padding_mask.to(
                device
            )

            logits = model(
                embeddings,
                padding_mask,
            )

            all_ids.extend(
                ids
            )

            all_logits.append(
                logits
                .detach()
                .cpu()
                .numpy()
            )

            all_labels.append(
                labels.numpy()
            )

    logits_array = np.concatenate(
        all_logits,
        axis=0,
    )

    labels_array = np.concatenate(
        all_labels,
        axis=0,
    )

    predictions = np.argmax(
        logits_array,
        axis=1,
    )

    accuracy = accuracy_score(
        labels_array,
        predictions,
    )

    return (
        all_ids,
        logits_array,
        labels_array,
        float(
            accuracy
        ),
    )


def save_score_table(
    ids: list[str],
    labels: np.ndarray,
    logits: np.ndarray,
    label_encoder: LabelEncoder,
    output_path: Path,
) -> None:
    """Save standardized class scores for multimodal integration."""

    if (
        len(ids)
        != len(labels)
        or len(ids)
        != len(logits)
    ):

        raise ValueError(
            "IDs, labels, and logits "
            "must contain the same number of rows."
        )

    score_columns = {
        f"score_{index}":
        logits[:, index]
        for index
        in range(
            logits.shape[1]
        )
    }

    output = pd.DataFrame(
        {
            "UniProtID":
            ids,

            "EC":
            label_encoder
            .inverse_transform(
                labels
            ),

            **score_columns,
        }
    )

    output.to_csv(
        output_path,
        index=False,
    )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Train a Transformer classifier "
            "on ProteinMPNN structural embeddings."
        )
    )

    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help=(
            "Master project metadata TSV "
            "containing UniProtID and EC."
        ),
    )

    parser.add_argument(
        "--embedding-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing "
            "*_embedding.npy files produced by "
            "extract_proteinmpnn_embeddings.py."
        ),
    )

    parser.add_argument(
        "--split-manifest",
        type=Path,
        required=True,
        help=(
            "Shared split manifest produced by "
            "create_split_manifest.py."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/structure"
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--num-heads",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--ff-dim",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--max-residues",
        type=int,
        default=None,
        help=(
            "Optional maximum number of residues "
            "per protein. Longer representations "
            "are truncated."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    if args.batch_size < 1:

        raise ValueError(
            "--batch-size must be at least 1."
        )

    if args.num_heads < 1:

        raise ValueError(
            "--num-heads must be at least 1."
        )

    if args.num_layers < 1:

        raise ValueError(
            "--num-layers must be at least 1."
        )

    if (
        args.max_residues is not None
        and args.max_residues < 1
    ):

        raise ValueError(
            "--max-residues must be positive."
        )

    set_seeds(
        args.seed
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        metadata,
        embeddings,
        manifest,
    ) = load_structure_data(
        metadata_path=(
            args.metadata
        ),
        embedding_dir=(
            args.embedding_dir
        ),
        split_manifest_path=(
            args.split_manifest
        ),
    )

    label_encoder = LabelEncoder()

    # Use exactly the same EC vocabulary as the other branches.
    label_encoder.fit(
        manifest[
            "EC"
        ].astype(str)
    )

    labels = (
        label_encoder
        .transform(
            metadata[
                "EC"
            ].astype(str)
        )
    )

    input_dim = int(
        embeddings[0]
        .shape[1]
    )

    if (
        input_dim
        % args.num_heads
        != 0
    ):

        raise ValueError(
            "ProteinMPNN feature dimension "
            f"({input_dim}) must be divisible by "
            f"--num-heads ({args.num_heads})."
        )

    train_dataset = build_dataset(
        metadata=metadata,
        embeddings=embeddings,
        split_name="train",
        labels=labels,
        max_residues=(
            args.max_residues
        ),
    )

    validation_dataset = build_dataset(
        metadata=metadata,
        embeddings=embeddings,
        split_name="validation",
        labels=labels,
        max_residues=(
            args.max_residues
        ),
    )

    test_dataset = build_dataset(
        metadata=metadata,
        embeddings=embeddings,
        split_name="test",
        labels=labels,
        max_residues=(
            args.max_residues
        ),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=(
            collate_structures
        ),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=(
            collate_structures
        ),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=(
            collate_structures
        ),
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    num_classes = len(
        label_encoder.classes_
    )

    print(
        f"Matched structure proteins: "
        f"{len(metadata)}"
    )

    print(
        f"Train: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation: "
        f"{len(validation_dataset)}"
    )

    print(
        f"Test: "
        f"{len(test_dataset)}"
    )

    print(
        f"ProteinMPNN feature dimension: "
        f"{input_dim}"
    )

    print(
        f"Shared EC vocabulary: "
        f"{num_classes}"
    )

    print(
        f"Device: "
        f"{device}"
    )

    model = StructureTransformer(
        input_dim=input_dim,
        num_classes=num_classes,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    ).to(
        device
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    best_validation_accuracy = -1.0
    best_state = None
    patience_counter = 0

    history = []

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        model.train()

        total_loss = 0.0
        train_predictions = []
        train_labels = []

        for (
            _,
            batch_embeddings,
            padding_mask,
            batch_labels,
        ) in train_loader:

            batch_embeddings = (
                batch_embeddings
                .to(device)
            )

            padding_mask = (
                padding_mask
                .to(device)
            )

            batch_labels = (
                batch_labels
                .to(device)
            )

            optimizer.zero_grad()

            logits = model(
                batch_embeddings,
                padding_mask,
            )

            loss = criterion(
                logits,
                batch_labels,
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
            )

            train_predictions.extend(
                logits
                .argmax(
                    dim=1
                )
                .detach()
                .cpu()
                .tolist()
            )

            train_labels.extend(
                batch_labels
                .detach()
                .cpu()
                .tolist()
            )

        train_accuracy = (
            accuracy_score(
                train_labels,
                train_predictions,
            )
        )

        (
            _,
            _,
            _,
            validation_accuracy,
        ) = evaluate(
            model=model,
            loader=validation_loader,
            device=device,
        )

        mean_train_loss = (
            total_loss
            / max(
                len(
                    train_loader
                ),
                1,
            )
        )

        history.append(
            {
                "epoch":
                epoch,

                "train_loss":
                float(
                    mean_train_loss
                ),

                "train_accuracy":
                float(
                    train_accuracy
                ),

                "validation_accuracy":
                float(
                    validation_accuracy
                ),
            }
        )

        print(
            f"Epoch {epoch:03d} | "
            f"loss={mean_train_loss:.4f} | "
            f"train_acc={train_accuracy:.4f} | "
            f"val_acc={validation_accuracy:.4f}"
        )

        if (
            validation_accuracy
            > best_validation_accuracy
        ):

            best_validation_accuracy = (
                validation_accuracy
            )

            best_state = copy.deepcopy(
                model.state_dict()
            )

            patience_counter = 0

        else:

            patience_counter += 1

            if (
                patience_counter
                >= args.patience
            ):

                print(
                    "Early stopping triggered."
                )

                break

    if best_state is None:

        raise RuntimeError(
            "Training did not produce "
            "a model checkpoint."
        )

    model.load_state_dict(
        best_state
    )

    (
        validation_ids,
        validation_logits,
        validation_labels,
        validation_accuracy,
    ) = evaluate(
        model=model,
        loader=validation_loader,
        device=device,
    )

    (
        test_ids,
        test_logits,
        test_labels,
        test_accuracy,
    ) = evaluate(
        model=model,
        loader=test_loader,
        device=device,
    )

    save_score_table(
        ids=validation_ids,
        labels=validation_labels,
        logits=validation_logits,
        label_encoder=label_encoder,
        output_path=(
            args.output_dir
            / "validation_scores.csv"
        ),
    )

    save_score_table(
        ids=test_ids,
        labels=test_labels,
        logits=test_logits,
        label_encoder=label_encoder,
        output_path=(
            args.output_dir
            / "test_scores.csv"
        ),
    )

    test_predictions = np.argmax(
        test_logits,
        axis=1,
    )

    pd.DataFrame(
        {
            "UniProtID":
            test_ids,

            "True_EC":
            label_encoder
            .inverse_transform(
                test_labels
            ),

            "Predicted_EC":
            label_encoder
            .inverse_transform(
                test_predictions
            ),
        }
    ).to_csv(
        args.output_dir
        / "test_predictions.csv",
        index=False,
    )

    pd.DataFrame(
        history
    ).to_csv(
        args.output_dir
        / "training_history.csv",
        index=False,
    )

    torch.save(
        best_state,
        args.output_dir
        / "structure_transformer.pt",
    )

    label_mapping = {
        str(index):
        label
        for index, label
        in enumerate(
            label_encoder.classes_
        )
    }

    (
        args.output_dir
        / "label_mapping.json"
    ).write_text(
        json.dumps(
            label_mapping,
            indent=2,
        ),
        encoding="utf-8",
    )

    metrics = {
        "manifest_proteins":
        len(
            manifest
        ),

        "matched_structure_proteins":
        len(
            metadata
        ),

        "train_records":
        len(
            train_dataset
        ),

        "validation_records":
        len(
            validation_dataset
        ),

        "test_records":
        len(
            test_dataset
        ),

        "input_dimension":
        input_dim,

        "num_ec_classes":
        num_classes,

        "num_heads":
        args.num_heads,

        "num_layers":
        args.num_layers,

        "feedforward_dimension":
        args.ff_dim,

        "dropout":
        args.dropout,

        "learning_rate":
        args.learning_rate,

        "batch_size":
        args.batch_size,

        "max_residues":
        args.max_residues,

        "epochs_requested":
        args.epochs,

        "epochs_completed":
        len(
            history
        ),

        "best_validation_accuracy":
        float(
            best_validation_accuracy
        ),

        "final_validation_accuracy":
        float(
            validation_accuracy
        ),

        "test_accuracy":
        float(
            test_accuracy
        ),

        "seed":
        args.seed,
    }

    (
        args.output_dir
        / "metrics.json"
    ).write_text(
        json.dumps(
            metrics,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nTraining complete."
    )

    print(
        f"Best validation accuracy: "
        f"{best_validation_accuracy:.4f}"
    )

    print(
        f"Held-out test accuracy: "
        f"{test_accuracy:.4f}"
    )

    print(
        f"Results saved to: "
        f"{args.output_dir}"
    )


if __name__ == "__main__":
    main()
