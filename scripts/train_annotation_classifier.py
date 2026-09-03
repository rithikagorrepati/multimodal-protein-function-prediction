"""Train an annotation-based EC classifier using BioBERT embeddings.

Workflow
--------
UniProt functional annotation
-> BioBERT sentence embedding
-> RNN classifier
-> Enzyme Commission (EC) prediction

All model branches use the same UniProt-level split manifest so proteins are
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
from torch.utils.data import DataLoader, Dataset


class AnnotationDataset(Dataset):
    """BioBERT protein-annotation embeddings paired with EC labels."""

    def __init__(
        self,
        ids: np.ndarray,
        embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> None:

        self.ids = ids

        self.embeddings = torch.tensor(
            embeddings,
            dtype=torch.float32,
        )

        self.labels = torch.tensor(
            labels,
            dtype=torch.long,
        )

    def __len__(self) -> int:

        return len(
            self.labels
        )

    def __getitem__(
        self,
        index: int,
    ):

        return (
            self.ids[index],
            self.embeddings[index],
            self.labels[index],
        )


class AnnotationRNNClassifier(nn.Module):
    """RNN classifier operating on BioBERT annotation representations."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:

        super().__init__()

        rnn_dropout = (
            dropout
            if num_layers > 1
            else 0.0
        )

        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=rnn_dropout,
        )

        self.classifier = nn.Linear(
            hidden_dim,
            num_classes,
        )

    def forward(
        self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:

        # Each BioBERT vector represents one protein annotation.
        # Add a sequence-length dimension of one so proteins remain
        # independent samples inside each batch.
        sequence = embeddings.unsqueeze(
            dim=1
        )

        _, hidden = self.rnn(
            sequence
        )

        final_hidden = hidden[-1]

        logits = self.classifier(
            final_hidden
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


def load_annotation_data(
    metadata_path: Path,
    embeddings_path: Path,
    split_manifest_path: Path,
):
    """Join BioBERT embeddings to the shared UniProt split manifest."""

    metadata = pd.read_csv(
        metadata_path
    )

    required_metadata_columns = {
        "EmbeddingRow",
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
            "Annotation metadata is missing required columns: "
            + ", ".join(
                sorted(
                    missing_metadata_columns
                )
            )
        )

    embeddings = np.load(
        embeddings_path
    )

    if embeddings.ndim != 2:

        raise ValueError(
            "BioBERT embeddings must have shape "
            "[proteins, embedding_dimension]."
        )

    if len(metadata) != len(embeddings):

        raise ValueError(
            "Annotation metadata and BioBERT embedding counts differ: "
            f"{len(metadata)} metadata rows vs "
            f"{len(embeddings)} embedding rows."
        )

    metadata["EmbeddingRow"] = (
        metadata[
            "EmbeddingRow"
        ]
        .astype(int)
    )

    expected_rows = np.arange(
        len(metadata)
    )

    if not np.array_equal(
        np.sort(
            metadata[
                "EmbeddingRow"
            ].to_numpy()
        ),
        expected_rows,
    ):

        raise ValueError(
            "EmbeddingRow must contain one unique index "
            "for every row in biobert_embeddings.npy."
        )

    metadata["UniProtID"] = (
        metadata[
            "UniProtID"
        ]
        .astype(str)
        .str.strip()
    )

    metadata["EC"] = (
        metadata["EC"]
        .astype(str)
        .str.strip()
    )

    if (
        metadata[
            "UniProtID"
        ]
        .duplicated()
        .any()
    ):

        raise ValueError(
            "Annotation metadata contains duplicate UniProt IDs."
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
        manifest["EC"]
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

    manifest_for_merge = (
        manifest[
            [
                "UniProtID",
                "EC",
                "split",
            ]
        ]
        .rename(
            columns={
                "EC":
                "manifest_EC"
            }
        )
    )

    combined = metadata.merge(
        manifest_for_merge,
        on="UniProtID",
        how="inner",
        validate="one_to_one",
    )

    if combined.empty:

        raise ValueError(
            "No annotation proteins matched "
            "the shared split manifest."
        )

    if not (
        combined["EC"]
        == combined[
            "manifest_EC"
        ]
    ).all():

        mismatches = combined[
            combined["EC"]
            != combined[
                "manifest_EC"
            ]
        ]

        raise ValueError(
            "EC labels differ between annotation metadata "
            "and the split manifest for "
            f"{len(mismatches)} proteins."
        )

    combined = combined.drop(
        columns=[
            "manifest_EC"
        ]
    )

    # Reconstruct the embedding matrix in the same order
    # as the merged metadata.
    embedding_rows = (
        combined[
            "EmbeddingRow"
        ]
        .to_numpy(
            dtype=int
        )
    )

    matched_embeddings = embeddings[
        embedding_rows
    ].astype(
        np.float32
    )

    label_encoder = LabelEncoder()

    # Every model branch uses the full shared EC vocabulary.
    label_encoder.fit(
        manifest["EC"]
    )

    combined["label"] = (
        label_encoder
        .transform(
            combined["EC"]
        )
    )

    return (
        combined,
        matched_embeddings,
        label_encoder,
        manifest,
    )


def build_loader(
    metadata: pd.DataFrame,
    embeddings: np.ndarray,
    split_name: str,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Create a DataLoader for one shared split."""

    mask = (
        metadata[
            "split"
        ]
        == split_name
    )

    selected_metadata = (
        metadata[
            mask
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    selected_embeddings = embeddings[
        mask.to_numpy()
    ]

    if selected_metadata.empty:

        raise ValueError(
            f"No annotation records found "
            f"for split '{split_name}'."
        )

    dataset = AnnotationDataset(
        ids=selected_metadata[
            "UniProtID"
        ].to_numpy(),
        embeddings=selected_embeddings,
        labels=selected_metadata[
            "label"
        ].to_numpy(),
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    return loader


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
):
    """Evaluate the model and return IDs, logits, labels, and accuracy."""

    model.eval()

    all_ids = []
    all_logits = []
    all_labels = []

    with torch.no_grad():

        for (
            batch_ids,
            batch_embeddings,
            batch_labels,
        ) in loader:

            batch_embeddings = (
                batch_embeddings
                .to(device)
            )

            logits = model(
                batch_embeddings
            )

            all_ids.extend(
                list(
                    batch_ids
                )
            )

            all_logits.append(
                logits
                .detach()
                .cpu()
                .numpy()
            )

            all_labels.append(
                batch_labels
                .numpy()
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
    """Save standardized model scores for multimodal integration."""

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
            "Train an RNN classifier "
            "on BioBERT protein-annotation embeddings."
        )
    )

    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help=(
            "annotation_metadata.csv produced by "
            "extract_biobert_embeddings.py."
        ),
    )

    parser.add_argument(
        "--embeddings",
        type=Path,
        required=True,
        help=(
            "biobert_embeddings.npy produced by "
            "extract_biobert_embeddings.py."
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
            "results/annotation"
        ),
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--layers",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=7e-4,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    if args.hidden_dim < 1:

        raise ValueError(
            "--hidden-dim must be positive."
        )

    if args.layers < 1:

        raise ValueError(
            "--layers must be at least 1."
        )

    if args.batch_size < 1:

        raise ValueError(
            "--batch-size must be at least 1."
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
        label_encoder,
        manifest,
    ) = load_annotation_data(
        metadata_path=(
            args.metadata
        ),
        embeddings_path=(
            args.embeddings
        ),
        split_manifest_path=(
            args.split_manifest
        ),
    )

    train_loader = build_loader(
        metadata=metadata,
        embeddings=embeddings,
        split_name="train",
        batch_size=args.batch_size,
        shuffle=True,
    )

    validation_loader = build_loader(
        metadata=metadata,
        embeddings=embeddings,
        split_name="validation",
        batch_size=args.batch_size,
        shuffle=False,
    )

    test_loader = build_loader(
        metadata=metadata,
        embeddings=embeddings,
        split_name="test",
        batch_size=args.batch_size,
        shuffle=False,
    )

    train_count = sum(
        metadata[
            "split"
        ]
        == "train"
    )

    validation_count = sum(
        metadata[
            "split"
        ]
        == "validation"
    )

    test_count = sum(
        metadata[
            "split"
        ]
        == "test"
    )

    input_dim = int(
        embeddings.shape[1]
    )

    num_classes = len(
        label_encoder.classes_
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Annotation proteins matched "
        f"to shared manifest: "
        f"{len(metadata)}"
    )

    print(
        f"Train: "
        f"{train_count}"
    )

    print(
        f"Validation: "
        f"{validation_count}"
    )

    print(
        f"Test: "
        f"{test_count}"
    )

    print(
        f"BioBERT embedding dimension: "
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

    model = AnnotationRNNClassifier(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        num_layers=args.layers,
        dropout=args.dropout,
    ).to(
        device
    )

    criterion = (
        nn.CrossEntropyLoss()
    )

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
            batch_labels,
        ) in train_loader:

            batch_embeddings = (
                batch_embeddings
                .to(device)
            )

            batch_labels = (
                batch_labels
                .to(device)
            )

            optimizer.zero_grad()

            logits = model(
                batch_embeddings
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

        train_accuracy = accuracy_score(
            train_labels,
            train_predictions,
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

    predictions = pd.DataFrame(
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
    )

    predictions.to_csv(
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
        / "annotation_rnn.pt",
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

        "matched_annotation_proteins":
        len(
            metadata
        ),

        "train_records":
        int(
            train_count
        ),

        "validation_records":
        int(
            validation_count
        ),

        "test_records":
        int(
            test_count
        ),

        "embedding_dimension":
        input_dim,

        "num_ec_classes":
        num_classes,

        "hidden_dimension":
        args.hidden_dim,

        "rnn_layers":
        args.layers,

        "dropout":
        args.dropout,

        "learning_rate":
        args.learning_rate,

        "batch_size":
        args.batch_size,

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
