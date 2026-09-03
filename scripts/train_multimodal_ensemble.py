"""Train a multimodal attention ensemble for EC-number prediction.

Workflow
--------
Sequence model scores
        +
Annotation model scores
        +
Structure model scores
        ↓
Multi-head attention over modalities
        ↓
Learned modality weights
        ↓
Weighted EC-class probabilities
        ↓
EC prediction

The three base-model branches must use the same UniProt-level split manifest
and the same EC label vocabulary.

The ensemble is developed using base-model predictions from the shared
validation partition and evaluated on the untouched shared test partition.
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
from torch.utils.data import DataLoader, Dataset


MODALITIES = (
    "annotation",
    "sequence",
    "structure",
)


class EnsembleDataset(Dataset):
    """Aligned probability vectors from three protein modalities."""

    def __init__(
        self,
        ids: np.ndarray,
        annotation: np.ndarray,
        sequence: np.ndarray,
        structure: np.ndarray,
        labels: np.ndarray,
    ) -> None:

        self.ids = ids

        self.annotation = torch.tensor(
            annotation,
            dtype=torch.float32,
        )

        self.sequence = torch.tensor(
            sequence,
            dtype=torch.float32,
        )

        self.structure = torch.tensor(
            structure,
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
            self.annotation[index],
            self.sequence[index],
            self.structure[index],
            self.labels[index],
        )


class AttentionFusionEnsemble(nn.Module):
    """Learn protein-specific weights for the three modalities."""

    def __init__(
        self,
        num_classes: int,
        embed_dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.2,
    ) -> None:

        super().__init__()

        self.annotation_projection = nn.Linear(
            num_classes,
            embed_dim,
        )

        self.sequence_projection = nn.Linear(
            num_classes,
            embed_dim,
        )

        self.structure_projection = nn.Linear(
            num_classes,
            embed_dim,
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(
            embed_dim
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.modality_gate = nn.Linear(
            embed_dim,
            1,
        )

    def forward(
        self,
        annotation_probs: torch.Tensor,
        sequence_probs: torch.Tensor,
        structure_probs: torch.Tensor,
    ):

        raw_probabilities = torch.stack(
            [
                annotation_probs,
                sequence_probs,
                structure_probs,
            ],
            dim=1,
        )

        modality_tokens = torch.stack(
            [
                self.annotation_projection(
                    annotation_probs
                ),
                self.sequence_projection(
                    sequence_probs
                ),
                self.structure_projection(
                    structure_probs
                ),
            ],
            dim=1,
        )

        attended, _ = self.attention(
            modality_tokens,
            modality_tokens,
            modality_tokens,
            need_weights=False,
        )

        modality_tokens = self.norm(
            modality_tokens
            + self.dropout(
                attended
            )
        )

        gate_logits = (
            self.modality_gate(
                modality_tokens
            )
            .squeeze(-1)
        )

        modality_weights = torch.softmax(
            gate_logits,
            dim=1,
        )

        fused_probabilities = (
            raw_probabilities
            * modality_weights
            .unsqueeze(-1)
        ).sum(
            dim=1
        )

        fused_probabilities = (
            fused_probabilities
            .clamp(
                min=1e-8
            )
        )

        return (
            fused_probabilities,
            modality_weights,
        )


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


def stable_softmax(
    scores: np.ndarray,
) -> np.ndarray:
    """Convert model logits into normalized class probabilities."""

    shifted = (
        scores
        - scores.max(
            axis=1,
            keepdims=True,
        )
    )

    exponentials = np.exp(
        shifted
    )

    denominator = exponentials.sum(
        axis=1,
        keepdims=True,
    )

    return (
        exponentials
        / np.clip(
            denominator,
            1e-12,
            None,
        )
    ).astype(
        np.float32
    )


def load_label_mapping(
    directory: Path,
) -> list[str]:
    """Read the EC index-to-label mapping from one model branch."""

    mapping_path = (
        directory
        / "label_mapping.json"
    )

    if not mapping_path.exists():

        raise FileNotFoundError(
            "Missing label mapping: "
            f"{mapping_path}"
        )

    mapping = json.loads(
        mapping_path.read_text(
            encoding="utf-8"
        )
    )

    indices = sorted(
        int(key)
        for key
        in mapping
    )

    expected = list(
        range(
            len(indices)
        )
    )

    if indices != expected:

        raise ValueError(
            f"{mapping_path} does not contain "
            "a contiguous class index mapping."
        )

    return [
        mapping[
            str(index)
        ]
        for index
        in indices
    ]


def validate_label_mappings(
    annotation_dir: Path,
    sequence_dir: Path,
    structure_dir: Path,
) -> list[str]:
    """Confirm that every base model uses the same EC vocabulary."""

    annotation_mapping = (
        load_label_mapping(
            annotation_dir
        )
    )

    sequence_mapping = (
        load_label_mapping(
            sequence_dir
        )
    )

    structure_mapping = (
        load_label_mapping(
            structure_dir
        )
    )

    if not (
        annotation_mapping
        == sequence_mapping
        == structure_mapping
    ):

        raise ValueError(
            "Base-model label mappings differ. "
            "All three models must use the same "
            "shared EC vocabulary."
        )

    return sequence_mapping


def load_score_table(
    path: Path,
):
    """Read one standardized base-model score table."""

    if not path.exists():

        raise FileNotFoundError(
            f"Score file not found: "
            f"{path}"
        )

    dataframe = pd.read_csv(
        path
    )

    required_columns = {
        "UniProtID",
        "EC",
    }

    missing_columns = (
        required_columns
        .difference(
            dataframe.columns
        )
    )

    if missing_columns:

        raise ValueError(
            f"{path} is missing required columns: "
            + ", ".join(
                sorted(
                    missing_columns
                )
            )
        )

    dataframe["UniProtID"] = (
        dataframe[
            "UniProtID"
        ]
        .astype(str)
        .str.strip()
    )

    dataframe["EC"] = (
        dataframe[
            "EC"
        ]
        .astype(str)
        .str.strip()
    )

    if (
        dataframe[
            "UniProtID"
        ]
        .duplicated()
        .any()
    ):

        raise ValueError(
            f"{path} contains duplicate "
            "UniProt IDs."
        )

    score_columns = [
        column
        for column
        in dataframe.columns
        if column.startswith(
            "score_"
        )
    ]

    if not score_columns:

        raise ValueError(
            f"{path} contains no "
            "score_* columns."
        )

    try:

        score_columns = sorted(
            score_columns,
            key=lambda column:
            int(
                column.split(
                    "_"
                )[-1]
            ),
        )

    except ValueError as exc:

        raise ValueError(
            f"{path} contains invalid "
            "score column names."
        ) from exc

    expected_columns = [
        f"score_{index}"
        for index
        in range(
            len(
                score_columns
            )
        )
    ]

    if (
        score_columns
        != expected_columns
    ):

        raise ValueError(
            f"{path} score columns "
            "are not contiguous."
        )

    return (
        dataframe,
        score_columns,
    )


def align_modalities(
    annotation_path: Path,
    sequence_path: Path,
    structure_path: Path,
    num_classes: int,
):
    """Align three base-model outputs by UniProt ID."""

    annotation_df, annotation_columns = (
        load_score_table(
            annotation_path
        )
    )

    sequence_df, sequence_columns = (
        load_score_table(
            sequence_path
        )
    )

    structure_df, structure_columns = (
        load_score_table(
            structure_path
        )
    )

    for (
        name,
        columns,
    ) in (
        (
            "annotation",
            annotation_columns,
        ),
        (
            "sequence",
            sequence_columns,
        ),
        (
            "structure",
            structure_columns,
        ),
    ):

        if len(
            columns
        ) != num_classes:

            raise ValueError(
                f"{name} model produced "
                f"{len(columns)} score columns, "
                f"but the shared EC vocabulary "
                f"contains {num_classes} classes."
            )

    annotation_df = (
        annotation_df
        .set_index(
            "UniProtID"
        )
    )

    sequence_df = (
        sequence_df
        .set_index(
            "UniProtID"
        )
    )

    structure_df = (
        structure_df
        .set_index(
            "UniProtID"
        )
    )

    common_ids = sorted(
        set(
            annotation_df.index
        )
        & set(
            sequence_df.index
        )
        & set(
            structure_df.index
        )
    )

    if not common_ids:

        raise ValueError(
            "No UniProt IDs are shared "
            "across all three model outputs."
        )

    annotation_labels = (
        annotation_df.loc[
            common_ids,
            "EC",
        ]
        .astype(str)
    )

    sequence_labels = (
        sequence_df.loc[
            common_ids,
            "EC",
        ]
        .astype(str)
    )

    structure_labels = (
        structure_df.loc[
            common_ids,
            "EC",
        ]
        .astype(str)
    )

    if not (
        np.array_equal(
            annotation_labels.to_numpy(),
            sequence_labels.to_numpy(),
        )
        and np.array_equal(
            annotation_labels.to_numpy(),
            structure_labels.to_numpy(),
        )
    ):

        raise ValueError(
            "EC labels do not agree "
            "across the three modalities."
        )

    annotation_logits = (
        annotation_df.loc[
            common_ids,
            annotation_columns,
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    sequence_logits = (
        sequence_df.loc[
            common_ids,
            sequence_columns,
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    structure_logits = (
        structure_df.loc[
            common_ids,
            structure_columns,
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    return {
        "ids":
        np.asarray(
            common_ids
        ),

        "ec":
        annotation_labels
        .to_numpy(),

        "annotation":
        stable_softmax(
            annotation_logits
        ),

        "sequence":
        stable_softmax(
            sequence_logits
        ),

        "structure":
        stable_softmax(
            structure_logits
        ),
    }


def create_dataset(
    data,
    indices: np.ndarray,
    labels: np.ndarray,
) -> EnsembleDataset:
    """Create an ensemble dataset from selected aligned proteins."""

    return EnsembleDataset(
        ids=data[
            "ids"
        ][indices],
        annotation=data[
            "annotation"
        ][indices],
        sequence=data[
            "sequence"
        ][indices],
        structure=data[
            "structure"
        ][indices],
        labels=labels[
            indices
        ],
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
):
    """Evaluate ensemble predictions and modality weights."""

    model.eval()

    all_ids = []
    all_probabilities = []
    all_weights = []
    all_labels = []

    with torch.no_grad():

        for (
            ids,
            annotation,
            sequence,
            structure,
            labels,
        ) in loader:

            annotation = (
                annotation.to(
                    device
                )
            )

            sequence = (
                sequence.to(
                    device
                )
            )

            structure = (
                structure.to(
                    device
                )
            )

            probabilities, weights = model(
                annotation,
                sequence,
                structure,
            )

            all_ids.extend(
                list(
                    ids
                )
            )

            all_probabilities.append(
                probabilities
                .cpu()
                .numpy()
            )

            all_weights.append(
                weights
                .cpu()
                .numpy()
            )

            all_labels.append(
                labels.numpy()
            )

    probabilities = np.concatenate(
        all_probabilities,
        axis=0,
    )

    modality_weights = np.concatenate(
        all_weights,
        axis=0,
    )

    labels = np.concatenate(
        all_labels,
        axis=0,
    )

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    return (
        all_ids,
        probabilities,
        modality_weights,
        labels,
        float(
            accuracy
        ),
    )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Train a multimodal attention "
            "ensemble for EC prediction."
        )
    )

    parser.add_argument(
        "--annotation-dir",
        type=Path,
        required=True,
        help=(
            "Annotation-model result directory."
        ),
    )

    parser.add_argument(
        "--sequence-dir",
        type=Path,
        required=True,
        help=(
            "Sequence-model result directory."
        ),
    )

    parser.add_argument(
        "--structure-dir",
        type=Path,
        required=True,
        help=(
            "Structure-model result directory."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/ensemble"
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
        default=32,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--embed-dim",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--num-heads",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--ensemble-validation-fraction",
        type=float,
        default=0.20,
        help=(
            "Fraction of the shared validation "
            "predictions reserved for ensemble "
            "model selection."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    if (
        args.embed_dim
        % args.num_heads
        != 0
    ):

        raise ValueError(
            "--embed-dim must be divisible "
            "by --num-heads."
        )

    if not (
        0.0
        < args.ensemble_validation_fraction
        < 1.0
    ):

        raise ValueError(
            "--ensemble-validation-fraction "
            "must be between 0 and 1."
        )

    set_seeds(
        args.seed
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    class_labels = (
        validate_label_mappings(
            annotation_dir=(
                args.annotation_dir
            ),
            sequence_dir=(
                args.sequence_dir
            ),
            structure_dir=(
                args.structure_dir
            ),
        )
    )

    num_classes = len(
        class_labels
    )

    label_to_index = {
        label:
        index
        for index, label
        in enumerate(
            class_labels
        )
    }

    development = align_modalities(
        annotation_path=(
            args.annotation_dir
            / "validation_scores.csv"
        ),
        sequence_path=(
            args.sequence_dir
            / "validation_scores.csv"
        ),
        structure_path=(
            args.structure_dir
            / "validation_scores.csv"
        ),
        num_classes=num_classes,
    )

    test = align_modalities(
        annotation_path=(
            args.annotation_dir
            / "test_scores.csv"
        ),
        sequence_path=(
            args.sequence_dir
            / "test_scores.csv"
        ),
        structure_path=(
            args.structure_dir
            / "test_scores.csv"
        ),
        num_classes=num_classes,
    )

    try:

        development_labels = np.asarray(
            [
                label_to_index[
                    label
                ]
                for label
                in development[
                    "ec"
                ]
            ],
            dtype=np.int64,
        )

        test_labels = np.asarray(
            [
                label_to_index[
                    label
                ]
                for label
                in test[
                    "ec"
                ]
            ],
            dtype=np.int64,
        )

    except KeyError as exc:

        raise ValueError(
            "A protein contains an EC label "
            "that is absent from the shared "
            "label mapping."
        ) from exc

    development_indices = np.arange(
        len(
            development_labels
        )
    )

    (
        ensemble_train_indices,
        ensemble_validation_indices,
    ) = train_test_split(
        development_indices,
        test_size=(
            args
            .ensemble_validation_fraction
        ),
        random_state=(
            args.seed
        ),
        shuffle=True,
    )

    train_dataset = create_dataset(
        data=development,
        indices=ensemble_train_indices,
        labels=development_labels,
    )

    validation_dataset = create_dataset(
        data=development,
        indices=(
            ensemble_validation_indices
        ),
        labels=development_labels,
    )

    test_indices = np.arange(
        len(
            test_labels
        )
    )

    test_dataset = create_dataset(
        data=test,
        indices=test_indices,
        labels=test_labels,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Shared EC classes: "
        f"{num_classes}"
    )

    print(
        f"Aligned development proteins: "
        f"{len(development_labels)}"
    )

    print(
        f"Ensemble train: "
        f"{len(train_dataset)}"
    )

    print(
        f"Ensemble validation: "
        f"{len(validation_dataset)}"
    )

    print(
        f"Aligned held-out test proteins: "
        f"{len(test_dataset)}"
    )

    print(
        f"Device: "
        f"{device}"
    )

    model = AttentionFusionEnsemble(
        num_classes=num_classes,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(
        device
    )

    criterion = nn.NLLLoss()

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
            annotation,
            sequence,
            structure,
            labels,
        ) in train_loader:

            annotation = annotation.to(
                device
            )

            sequence = sequence.to(
                device
            )

            structure = structure.to(
                device
            )

            labels = labels.to(
                device
            )

            optimizer.zero_grad()

            probabilities, _ = model(
                annotation,
                sequence,
                structure,
            )

            loss = criterion(
                torch.log(
                    probabilities
                ),
                labels,
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
            )

            train_predictions.extend(
                probabilities
                .argmax(
                    dim=1
                )
                .detach()
                .cpu()
                .tolist()
            )

            train_labels.extend(
                labels
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
            _,
            validation_accuracy,
        ) = evaluate(
            model=model,
            loader=validation_loader,
            device=device,
        )

        mean_loss = (
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
                    mean_loss
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
            f"loss={mean_loss:.4f} | "
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
            "an ensemble checkpoint."
        )

    model.load_state_dict(
        best_state
    )

    (
        test_ids,
        test_probabilities,
        test_weights,
        final_test_labels,
        test_accuracy,
    ) = evaluate(
        model=model,
        loader=test_loader,
        device=device,
    )

    test_predictions = np.argmax(
        test_probabilities,
        axis=1,
    )

    # Simple equal-weight ensemble for comparison.
    average_probabilities = (
        test[
            "annotation"
        ]
        + test[
            "sequence"
        ]
        + test[
            "structure"
        ]
    ) / 3.0

    average_predictions = np.argmax(
        average_probabilities,
        axis=1,
    )

    average_accuracy = accuracy_score(
        test_labels,
        average_predictions,
    )

    predictions = pd.DataFrame(
        {
            "UniProtID":
            test_ids,

            "True_EC":
            [
                class_labels[
                    index
                ]
                for index
                in final_test_labels
            ],

            "Predicted_EC":
            [
                class_labels[
                    index
                ]
                for index
                in test_predictions
            ],

            "AnnotationWeight":
            test_weights[
                :,
                0,
            ],

            "SequenceWeight":
            test_weights[
                :,
                1,
            ],

            "StructureWeight":
            test_weights[
                :,
                2,
            ],
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
        / "multimodal_attention.pt",
    )

    mean_weights = (
        test_weights.mean(
            axis=0
        )
    )

    metrics = {
        "num_ec_classes":
        num_classes,

        "aligned_development_proteins":
        len(
            development_labels
        ),

        "ensemble_train_records":
        len(
            train_dataset
        ),

        "ensemble_validation_records":
        len(
            validation_dataset
        ),

        "aligned_test_proteins":
        len(
            test_dataset
        ),

        "embed_dimension":
        args.embed_dim,

        "attention_heads":
        args.num_heads,

        "best_validation_accuracy":
        float(
            best_validation_accuracy
        ),

        "test_accuracy":
        float(
            test_accuracy
        ),

        "equal_weight_test_accuracy":
        float(
            average_accuracy
        ),

        "mean_annotation_weight":
        float(
            mean_weights[0]
        ),

        "mean_sequence_weight":
        float(
            mean_weights[1]
        ),

        "mean_structure_weight":
        float(
            mean_weights[2]
        ),

        "epochs_completed":
        len(
            history
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
        "\nEnsemble training complete."
    )

    print(
        f"Best ensemble validation accuracy: "
        f"{best_validation_accuracy:.4f}"
    )

    print(
        f"Attention ensemble test accuracy: "
        f"{test_accuracy:.4f}"
    )

    print(
        f"Equal-weight ensemble test accuracy: "
        f"{average_accuracy:.4f}"
    )

    print(
        "\nMean learned modality weights:"
    )

    print(
        f"Annotation: "
        f"{mean_weights[0]:.3f}"
    )

    print(
        f"Sequence: "
        f"{mean_weights[1]:.3f}"
    )

    print(
        f"Structure: "
        f"{mean_weights[2]:.3f}"
    )


if __name__ == "__main__":
    main()
