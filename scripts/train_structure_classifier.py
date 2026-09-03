"""Train a Transformer classifier on ProteinMPNN structural embeddings.

Workflow:
AlphaFold PDB -> ProteinMPNN per-residue embeddings -> Transformer -> EC class

The original course implementation flattened each structure before applying a
Transformer. This cleaned implementation keeps the residue dimension intact so
self-attention operates within each protein rather than across proteins in a
batch.
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
from sklearn.model_selection import train_test_split
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
        return len(self.ids)

    def __getitem__(self, index: int):
        embedding = self.embeddings[index]

        if self.max_residues is not None:
            embedding = embedding[: self.max_residues]

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


def collate_structures(batch):
    """Pad variable-length residue embeddings within one batch."""

    ids, embeddings, labels = zip(*batch)

    lengths = torch.tensor(
        [
            embedding.shape[0]
            for embedding in embeddings
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
    ).unsqueeze(0)

    padding_mask = (
        positions
        >= lengths.unsqueeze(1)
    )

    return (
        list(ids),
        padded,
        padding_mask,
        torch.stack(labels),
    )


class StructureTransformer(nn.Module):
    """Transformer over per-residue ProteinMPNN representations."""

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

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
        )

        self.encoder = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
            )
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

        valid = (
            (~padding_mask)
            .unsqueeze(-1)
            .to(encoded.dtype)
        )

        pooled = (
            (encoded * valid).sum(dim=1)
            / valid.sum(dim=1).clamp(
                min=1.0
            )
        )

        return self.classifier(
            pooled
        )


def set_seeds(seed: int) -> None:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


def load_structure_data(
    metadata_path: Path,
    embedding_dir: Path,
):
    """Join ProteinMPNN embeddings with EC labels."""

    metadata = pd.read_csv(
        metadata_path,
        sep="\t",
    )

    required = {
        "UniProtID",
        "EC",
    }

    missing = required.difference(
        metadata.columns
    )

    if missing:
        raise ValueError(
            "Metadata is missing required columns: "
            + ", ".join(
                sorted(missing)
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
        metadata["UniProtID"]
        .astype(str)
        .str.strip()
    )

    metadata["EC"] = (
        metadata["EC"]
        .astype(str)
        .str.strip()
    )

    # The original project used one EC label
    # per protein record.
    metadata = (
        metadata
        .drop_duplicates(
            subset=["UniProtID"],
            keep="first",
        )
    )

    ec_by_id = (
        metadata
        .set_index("UniProtID")["EC"]
        .to_dict()
    )

    ids = []
    embeddings = []
    ec_labels = []

    for path in sorted(
        embedding_dir.glob(
            "*_embedding.npy"
        )
    ):

        uniprot_id = (
            path.name.removesuffix(
                "_embedding.npy"
            )
        )

        if uniprot_id not in ec_by_id:
            continue

        array = np.load(
            path
        )

        if array.ndim != 2:
            raise ValueError(
                f"{path.name} must have "
                "shape [residues, features]."
            )

        ids.append(
            uniprot_id
        )

        embeddings.append(
            array.astype(
                np.float32
            )
        )

        ec_labels.append(
            ec_by_id[
                uniprot_id
            ]
        )

    if not ids:
        raise ValueError(
            "No structure embeddings could "
            "be matched to EC labels."
        )

    return (
        metadata,
        ids,
        embeddings,
        ec_labels,
    )


def evaluate(
    model,
    loader,
    device,
):

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

            embeddings = (
                embeddings.to(device)
            )

            padding_mask = (
                padding_mask.to(device)
            )

            logits = model(
                embeddings,
                padding_mask,
            )

            all_ids.extend(
                ids
            )

            all_logits.append(
                logits.cpu().numpy()
            )

            all_labels.append(
                labels.numpy()
            )

    logits = np.concatenate(
        all_logits,
        axis=0,
    )

    labels = np.concatenate(
        all_labels,
        axis=0,
    )

    predictions = logits.argmax(
        axis=1
    )

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    return (
        accuracy,
        all_ids,
        logits,
        labels,
    )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Train an EC classifier on "
            "ProteinMPNN structural embeddings."
        )
    )

    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help=(
            "Master project metadata TSV."
        ),
    )

    parser.add_argument(
        "--embedding-dir",
        type=Path,
        required=True,
        help=(
            "Directory created by "
            "extract_proteinmpnn_embeddings.py."
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
        "--max-residues",
        type=int,
        default=None,
        help=(
            "Optional maximum residues "
            "per protein."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    set_seeds(
        args.seed
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        master_metadata,
        ids,
        embeddings,
        ec_labels,
    ) = load_structure_data(
        args.metadata,
        args.embedding_dir,
    )

    # Use the master project EC vocabulary.
    label_encoder = (
        LabelEncoder()
    )

    label_encoder.fit(
        master_metadata[
            "EC"
        ].astype(str)
    )

    labels = (
        label_encoder
        .transform(
            ec_labels
        )
    )

    indices = np.arange(
        len(ids)
    )

    # Preserve the original 70/10/20 split.
    (
        train_indices,
        test_indices,
    ) = train_test_split(
        indices,
        test_size=0.20,
        random_state=0,
    )

    (
        train_indices,
        val_indices,
    ) = train_test_split(
        train_indices,
        test_size=0.125,
        random_state=64,
    )

    def select(
        index_array,
    ):

        return StructureDataset(
            ids=[
                ids[i]
                for i
                in index_array
            ],
            embeddings=[
                embeddings[i]
                for i
                in index_array
            ],
            labels=labels[
                index_array
            ],
            max_residues=(
                args.max_residues
            ),
        )

    train_dataset = select(
        train_indices
    )

    val_dataset = select(
        val_indices
    )

    test_dataset = select(
        test_indices
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_structures,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_structures,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_structures,
    )

    input_dim = (
        embeddings[0]
        .shape[1]
    )

    if (
        input_dim
        % args.num_heads
        != 0
    ):
        raise ValueError(
            "Embedding dimension must "
            "be divisible by the number "
            "of attention heads."
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    print(
        f"Matched structures: "
        f"{len(ids)}"
    )

    print(
        f"Train: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation: "
        f"{len(val_dataset)}"
    )

    print(
        f"Test: "
        f"{len(test_dataset)}"
    )

    print(
        f"EC classes: "
        f"{len(label_encoder.classes_)}"
    )

    print(
        f"ProteinMPNN feature "
        f"dimension: {input_dim}"
    )

    model = StructureTransformer(
        input_dim=input_dim,
        num_classes=len(
            label_encoder.classes_
        ),
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
    ).to(device)

    criterion = (
        nn.CrossEntropyLoss()
    )

    optimizer = (
        torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
        )
    )

    best_val_accuracy = -1.0
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
                batch_embeddings.to(
                    device
                )
            )

            padding_mask = (
                padding_mask.to(
                    device
                )
            )

            batch_labels = (
                batch_labels.to(
                    device
                )
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
                logits.argmax(dim=1)
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
            val_accuracy,
            _,
            _,
            _,
        ) = evaluate(
            model,
            val_loader,
            device,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": (
                    total_loss
                    / max(
                        len(train_loader),
                        1,
                    )
                ),
                "train_accuracy": float(
                    train_accuracy
                ),
                "validation_accuracy": float(
                    val_accuracy
                ),
            }
        )

        print(
            f"Epoch {epoch:03d} | "
            f"loss="
            f"{history[-1]['train_loss']:.4f} | "
            f"train_acc="
            f"{train_accuracy:.4f} | "
            f"val_acc="
            f"{val_accuracy:.4f}"
        )

        if (
            val_accuracy
            > best_val_accuracy
        ):

            best_val_accuracy = (
                val_accuracy
            )

            best_state = (
                copy.deepcopy(
                    model.state_dict()
                )
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
        test_accuracy,
        test_ids,
        test_logits,
        test_labels,
    ) = evaluate(
        model,
        test_loader,
        device,
    )

    test_predictions = (
        test_logits.argmax(
            axis=1
        )
    )

    pd.DataFrame(
        {
            "UniProtID": (
                test_ids
            ),
            "True_EC": (
                label_encoder
                .inverse_transform(
                    test_labels
                )
            ),
            "Predicted_EC": (
                label_encoder
                .inverse_transform(
                    test_predictions
                )
            ),
        }
    ).to_csv(
        args.output_dir
        / "test_predictions.csv",
        index=False,
    )

    np.save(
        args.output_dir
        / "test_logits.npy",
        test_logits.astype(
            np.float32
        ),
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
        str(index): label
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
        "matched_structures": len(
            ids
        ),
        "train_records": len(
            train_dataset
        ),
        "validation_records": len(
            val_dataset
        ),
        "test_records": len(
            test_dataset
        ),
        "num_ec_classes": len(
            label_encoder.classes_
        ),
        "input_dimension": int(
            input_dim
        ),
        "num_heads": (
            args.num_heads
        ),
        "num_layers": (
            args.num_layers
        ),
        "feedforward_dimension": (
            args.ff_dim
        ),
        "max_residues": (
            args.max_residues
        ),
        "best_validation_accuracy": float(
            best_val_accuracy
        ),
        "test_accuracy": float(
            test_accuracy
        ),
        "seed": args.seed,
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
        f"Best validation accuracy: "
        f"{best_val_accuracy:.4f}"
    )

    print(
        f"Held-out test accuracy: "
        f"{test_accuracy:.4f}"
    )


if __name__ == "__main__":
    main()
