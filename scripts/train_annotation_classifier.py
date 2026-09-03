"""Train an RNN classifier on BioBERT protein-annotation embeddings.

Workflow:
UniProt functional annotation -> BioBERT sentence embedding -> RNN -> EC class

The original project used a PyTorch RNN classifier on BioBERT sentence vectors.
This cleaned implementation preserves that model family while ensuring each
protein embedding is treated as an independent sample.
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
from torch.utils.data import DataLoader, Dataset


class AnnotationDataset(Dataset):
    """Protein annotation embeddings paired with EC labels."""

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
        return len(self.labels)

    def __getitem__(self, index: int):
        return (
            self.ids[index],
            self.embeddings[index],
            self.labels[index],
        )


class RNNClassifier(nn.Module):
    """RNN classifier for fixed-length BioBERT sentence embeddings."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 1,
    ) -> None:
        super().__init__()

        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.classifier = nn.Linear(
            hidden_dim,
            num_classes,
        )

    def forward(
        self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:

        # One BioBERT vector represents one protein annotation.
        # Add a length-one sequence dimension so proteins in the
        # same batch remain independent samples.
        sequence = embeddings.unsqueeze(1)

        _, hidden = self.rnn(sequence)

        return self.classifier(
            hidden[-1]
        )


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(
    ids: np.ndarray,
    embeddings: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:

    dataset = AnnotationDataset(
        ids,
        embeddings,
        labels,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
):
    """Evaluate a classifier and return logits and labels."""

    model.eval()

    all_ids = []
    all_logits = []
    all_labels = []

    with torch.no_grad():

        for ids, embeddings, labels in loader:

            embeddings = embeddings.to(device)

            logits = model(
                embeddings
            )

            all_ids.extend(
                list(ids)
            )

            all_logits.append(
                logits.cpu().numpy()
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

    predictions = logits_array.argmax(
        axis=1
    )

    accuracy = accuracy_score(
        labels_array,
        predictions,
    )

    return (
        accuracy,
        all_ids,
        logits_array,
        labels_array,
    )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Train an RNN classifier "
            "on BioBERT annotation embeddings."
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

    set_seeds(
        args.seed
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = pd.read_csv(
        args.metadata
    )

    embeddings = np.load(
        args.embeddings
    )

    required_columns = {
        "UniProtID",
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

    if len(metadata) != len(embeddings):

        raise ValueError(
            "Metadata and embedding counts do not match: "
            f"{len(metadata)} metadata rows vs "
            f"{len(embeddings)} embeddings."
        )

    label_encoder = LabelEncoder()

    labels = label_encoder.fit_transform(
        metadata["EC"].astype(str)
    )

    ids = (
        metadata["UniProtID"]
        .astype(str)
        .to_numpy()
    )

    indices = np.arange(
        len(metadata)
    )

    # Match the split used in the annotation notebook:
    # 80% train pool / 20% test, followed by
    # an 80% / 20% train-validation split.
    train_indices, test_indices = train_test_split(
        indices,
        test_size=0.20,
        random_state=0,
    )

    train_indices, val_indices = train_test_split(
        train_indices,
        test_size=0.20,
        random_state=16,
    )

    print(
        f"Train: {len(train_indices)}"
    )

    print(
        f"Validation: {len(val_indices)}"
    )

    print(
        f"Test: {len(test_indices)}"
    )

    print(
        f"EC classes: "
        f"{len(label_encoder.classes_)}"
    )

    print(
        f"Embedding dimension: "
        f"{embeddings.shape[1]}"
    )

    train_loader = make_loader(
        ids[train_indices],
        embeddings[train_indices],
        labels[train_indices],
        args.batch_size,
        shuffle=True,
    )

    val_loader = make_loader(
        ids[val_indices],
        embeddings[val_indices],
        labels[val_indices],
        args.batch_size,
        shuffle=False,
    )

    test_loader = make_loader(
        ids[test_indices],
        embeddings[test_indices],
        labels[test_indices],
        args.batch_size,
        shuffle=False,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    model = RNNClassifier(
        input_dim=embeddings.shape[1],
        hidden_dim=args.hidden_dim,
        num_classes=len(
            label_encoder.classes_
        ),
        num_layers=args.layers,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
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
            batch_labels,
        ) in train_loader:

            batch_embeddings = (
                batch_embeddings.to(device)
            )

            batch_labels = (
                batch_labels.to(device)
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

            total_loss += loss.item()

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

        train_accuracy = accuracy_score(
            train_labels,
            train_predictions,
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
            f"loss={history[-1]['train_loss']:.4f} | "
            f"train_acc={train_accuracy:.4f} | "
            f"val_acc={val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = (
                val_accuracy
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
        test_logits.argmax(axis=1)
    )

    predictions = pd.DataFrame(
        {
            "UniProtID": test_ids,
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
    )

    predictions.to_csv(
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
        / "annotation_rnn.pt",
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
        "train_records": len(
            train_indices
        ),
        "validation_records": len(
            val_indices
        ),
        "test_records": len(
            test_indices
        ),
        "num_ec_classes": len(
            label_encoder.classes_
        ),
        "embedding_dimension": int(
            embeddings.shape[1]
        ),
        "hidden_dimension": (
            args.hidden_dim
        ),
        "rnn_layers": args.layers,
        "learning_rate": (
            args.learning_rate
        ),
        "batch_size": (
            args.batch_size
        ),
        "patience": args.patience,
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

    print(
        f"Results saved to: "
        f"{args.output_dir}"
    )


if __name__ == "__main__":
    main()
