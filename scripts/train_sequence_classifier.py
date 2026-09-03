"""Train the ProtT5 sequence classifier for EC-number prediction.

Workflow
--------
UniProt protein sequence
-> ProtT5 encoder
-> LoRA parameter-efficient fine-tuning
-> multiclass Enzyme Commission (EC) prediction

All model branches use a shared UniProt-level split manifest so that the same
protein cannot be assigned to different train/validation/test partitions across
modalities.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from datasets import Dataset
from peft import LoraConfig, inject_adapter_in_model
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from transformers import (
    T5EncoderModel,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.modeling_outputs import SequenceClassifierOutput


CHECKPOINT = "Rostlab/prot_t5_xl_uniref50"

UNCOMMON_AMINO_ACIDS = re.compile(
    r"[OBUZJ]"
)


class ProtT5Classifier(nn.Module):
    """ProtT5 encoder with a trainable classification head."""

    def __init__(
        self,
        encoder: T5EncoderModel,
        num_labels: int,
        dropout: float = 0.2,
    ) -> None:

        super().__init__()

        self.encoder = encoder

        hidden_size = (
            encoder.config.d_model
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.dense = nn.Linear(
            hidden_size,
            hidden_size,
        )

        self.out_proj = nn.Linear(
            hidden_size,
            num_labels,
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs,
    ):

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        hidden_state = (
            outputs.last_hidden_state
        )

        # Mean-pool only real amino-acid tokens.
        if attention_mask is not None:

            mask = (
                attention_mask
                .unsqueeze(-1)
                .to(
                    hidden_state.dtype
                )
            )

            pooled = (
                (
                    hidden_state
                    * mask
                )
                .sum(dim=1)
                / mask
                .sum(dim=1)
                .clamp(min=1.0)
            )

        else:

            pooled = (
                hidden_state.mean(
                    dim=1
                )
            )

        pooled = self.dropout(
            pooled
        )

        pooled = torch.tanh(
            self.dense(
                pooled
            )
        )

        pooled = self.dropout(
            pooled
        )

        logits = self.out_proj(
            pooled
        )

        loss = None

        if labels is not None:

            loss = (
                nn.CrossEntropyLoss()(
                    logits,
                    labels,
                )
            )

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
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

    set_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


def prepare_sequences(
    sequences: pd.Series,
) -> list[str]:
    """Prepare amino-acid sequences for ProtT5 tokenization."""

    cleaned = (
        sequences
        .astype(str)
        .map(
            lambda sequence:
            UNCOMMON_AMINO_ACIDS.sub(
                "X",
                sequence
                .strip()
                .upper(),
            )
        )
    )

    # ProtT5 expects amino acids separated by spaces.
    return [
        " ".join(
            sequence
        )
        for sequence
        in cleaned
    ]


def tokenize_dataset(
    tokenizer: T5Tokenizer,
    sequences: list[str],
    labels: list[int],
) -> Dataset:
    """Tokenize protein sequences and attach EC labels."""

    tokenized = tokenizer(
        sequences,
        max_length=1024,
        padding=True,
        truncation=True,
    )

    dataset = Dataset.from_dict(
        tokenized
    )

    dataset = dataset.add_column(
        "labels",
        labels,
    )

    return dataset


def build_model(
    num_labels: int,
) -> tuple[
    nn.Module,
    T5Tokenizer,
]:
    """Load ProtT5 and inject LoRA adapters."""

    tokenizer = (
        T5Tokenizer
        .from_pretrained(
            CHECKPOINT,
            do_lower_case=False,
        )
    )

    encoder = (
        T5EncoderModel
        .from_pretrained(
            CHECKPOINT
        )
    )

    model = ProtT5Classifier(
        encoder=encoder,
        num_labels=num_labels,
    )

    lora_config = LoraConfig(
        r=4,
        lora_alpha=1,
        bias="all",
        target_modules=[
            "q",
            "k",
            "v",
            "o",
        ],
    )

    model = (
        inject_adapter_in_model(
            lora_config,
            model,
        )
    )

    # Keep the classification head trainable.
    for parameter in (
        model.dense.parameters()
    ):
        parameter.requires_grad = True

    for parameter in (
        model.out_proj.parameters()
    ):
        parameter.requires_grad = True

    trainable_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    total_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    print(
        "Trainable parameters: "
        f"{trainable_parameters:,} "
        f"/ {total_parameters:,}"
    )

    return (
        model,
        tokenizer,
    )


def load_split_data(
    data_path: Path,
    split_manifest_path: Path,
):
    """Join the sequence dataset to the shared protein split manifest."""

    data = pd.read_csv(
        data_path
    )

    required_data_columns = {
        "UniProtID",
        "Sequence",
        "EC",
    }

    missing_data_columns = (
        required_data_columns
        .difference(
            data.columns
        )
    )

    if missing_data_columns:

        raise ValueError(
            "Sequence dataset is missing required columns: "
            + ", ".join(
                sorted(
                    missing_data_columns
                )
            )
        )

    data = (
        data
        .dropna(
            subset=[
                "UniProtID",
                "Sequence",
                "EC",
            ]
        )
        .copy()
    )

    data["UniProtID"] = (
        data["UniProtID"]
        .astype(str)
        .str.strip()
    )

    data["EC"] = (
        data["EC"]
        .astype(str)
        .str.strip()
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
        manifest["UniProtID"]
        .astype(str)
        .str.strip()
    )

    manifest["EC"] = (
        manifest["EC"]
        .astype(str)
        .str.strip()
    )

    manifest["split"] = (
        manifest["split"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    if (
        manifest["UniProtID"]
        .duplicated()
        .any()
    ):

        raise ValueError(
            "Split manifest contains "
            "duplicate UniProt IDs."
        )

    valid_splits = {
        "train",
        "validation",
        "test",
    }

    unexpected_splits = (
        set(
            manifest["split"]
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

    data = data.merge(
        manifest_for_merge,
        on="UniProtID",
        how="inner",
        validate="one_to_one",
    )

    if not (
        data["EC"]
        == data["manifest_EC"]
    ).all():

        mismatches = data[
            data["EC"]
            != data["manifest_EC"]
        ]

        raise ValueError(
            "EC labels differ between "
            "the sequence dataset and "
            "shared split manifest for "
            f"{len(mismatches)} proteins."
        )

    data = data.drop(
        columns=[
            "manifest_EC"
        ]
    )

    # Every modality uses the same class vocabulary.
    label_encoder = (
        LabelEncoder()
    )

    label_encoder.fit(
        manifest["EC"]
    )

    data["label"] = (
        label_encoder
        .transform(
            data["EC"]
        )
    )

    train_df = (
        data[
            data["split"]
            == "train"
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    val_df = (
        data[
            data["split"]
            == "validation"
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    test_df = (
        data[
            data["split"]
            == "test"
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    if train_df.empty:

        raise ValueError(
            "No sequence records "
            "matched the training split."
        )

    if val_df.empty:

        raise ValueError(
            "No sequence records "
            "matched the validation split."
        )

    if test_df.empty:

        raise ValueError(
            "No sequence records "
            "matched the test split."
        )

    return (
        train_df,
        val_df,
        test_df,
        label_encoder,
        manifest,
    )


def save_trainable_parameters(
    model: nn.Module,
    output_path: Path,
) -> None:
    """Save parameters that were updated during fine-tuning."""

    trainable_state = {
        name:
        parameter
        .detach()
        .cpu()
        for name, parameter
        in model.named_parameters()
        if parameter.requires_grad
    }

    torch.save(
        trainable_state,
        output_path,
    )


def save_score_table(
    dataframe: pd.DataFrame,
    logits: np.ndarray,
    output_path: Path,
) -> None:
    """Save model class scores with protein identifiers."""

    if len(dataframe) != len(logits):

        raise ValueError(
            "Prediction count does not "
            "match dataframe length."
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
            dataframe[
                "UniProtID"
            ].to_numpy(),

            "EC":
            dataframe[
                "EC"
            ].to_numpy(),

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
            "Fine-tune ProtT5 with LoRA "
            "for EC-number classification."
        )
    )

    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help=(
            "Sequence dataset CSV containing "
            "UniProtID, Sequence, and EC."
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
            "results/sequence"
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
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
        train_df,
        val_df,
        test_df,
        label_encoder,
        manifest,
    ) = load_split_data(
        data_path=args.data,
        split_manifest_path=(
            args.split_manifest
        ),
    )

    print(
        "Sequence records matched "
        "to shared manifest: "
        f"{len(train_df) + len(val_df) + len(test_df)}"
    )

    print(
        f"Train: "
        f"{len(train_df)}"
    )

    print(
        f"Validation: "
        f"{len(val_df)}"
    )

    print(
        f"Test: "
        f"{len(test_df)}"
    )

    print(
        "Shared EC vocabulary: "
        f"{len(label_encoder.classes_)}"
    )

    train_classes = set(
        train_df[
            "label"
        ].unique()
    )

    all_classes = set(
        range(
            len(
                label_encoder.classes_
            )
        )
    )

    unseen_train_classes = (
        all_classes
        - train_classes
    )

    if unseen_train_classes:

        print(
            "Warning: "
            f"{len(unseen_train_classes)} "
            "EC classes are absent from "
            "the sequence training subset."
        )

    model, tokenizer = (
        build_model(
            num_labels=len(
                label_encoder.classes_
            )
        )
    )

    train_dataset = (
        tokenize_dataset(
            tokenizer,
            prepare_sequences(
                train_df[
                    "Sequence"
                ]
            ),
            train_df[
                "label"
            ].tolist(),
        )
    )

    val_dataset = (
        tokenize_dataset(
            tokenizer,
            prepare_sequences(
                val_df[
                    "Sequence"
                ]
            ),
            val_df[
                "label"
            ].tolist(),
        )
    )

    test_dataset = (
        tokenize_dataset(
            tokenizer,
            prepare_sequences(
                test_df[
                    "Sequence"
                ]
            ),
            test_df[
                "label"
            ].tolist(),
        )
    )

    def compute_metrics(
        eval_prediction,
    ):

        logits, labels = (
            eval_prediction
        )

        predictions = (
            np.argmax(
                logits,
                axis=1,
            )
        )

        return {
            "accuracy":
            accuracy_score(
                labels,
                predictions,
            )
        }

    training_args = (
        TrainingArguments(
            output_dir=str(
                args.output_dir
                / "trainer"
            ),
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="no",
            learning_rate=3e-4,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=16,
            gradient_accumulation_steps=2,
            num_train_epochs=args.epochs,
            seed=args.seed,
            fp16=torch.cuda.is_available(),
            report_to=[],
        )
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Validation scores are useful for diagnostics.
    validation_output = (
        trainer.predict(
            val_dataset
        )
    )

    validation_logits = (
        validation_output
        .predictions
    )

    save_score_table(
        dataframe=val_df,
        logits=validation_logits,
        output_path=(
            args.output_dir
            / "validation_scores.csv"
        ),
    )

    # Final evaluation is performed only once
    # on the shared held-out test split.
    test_output = (
        trainer.predict(
            test_dataset
        )
    )

    test_logits = (
        test_output.predictions
    )

    test_predictions = (
        np.argmax(
            test_logits,
            axis=1,
        )
    )

    test_accuracy = (
        accuracy_score(
            test_df[
                "label"
            ],
            test_predictions,
        )
    )

    predicted_ec = (
        label_encoder
        .inverse_transform(
            test_predictions
        )
    )

    predictions_df = (
        pd.DataFrame(
            {
                "UniProtID":
                test_df[
                    "UniProtID"
                ].to_numpy(),

                "True_EC":
                test_df[
                    "EC"
                ].to_numpy(),

                "Predicted_EC":
                predicted_ec,
            }
        )
    )

    predictions_df.to_csv(
        args.output_dir
        / "test_predictions.csv",
        index=False,
    )

    save_score_table(
        dataframe=test_df,
        logits=test_logits,
        output_path=(
            args.output_dir
            / "test_scores.csv"
        ),
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
        "checkpoint":
        CHECKPOINT,

        "manifest_proteins":
        len(
            manifest
        ),

        "sequence_train_records":
        len(
            train_df
        ),

        "sequence_validation_records":
        len(
            val_df
        ),

        "sequence_test_records":
        len(
            test_df
        ),

        "num_ec_classes":
        len(
            label_encoder.classes_
        ),

        "test_accuracy":
        float(
            test_accuracy
        ),

        "epochs":
        args.epochs,

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

    save_trainable_parameters(
        model,
        args.output_dir
        / (
            "prot_t5_lora_"
            "trainable_parameters.pt"
        ),
    )

    print(
        "\nTraining complete."
    )

    print(
        "Held-out test accuracy: "
        f"{test_accuracy:.4f}"
    )

    print(
        "Results saved to: "
        f"{args.output_dir}"
    )


if __name__ == "__main__":
    main()
