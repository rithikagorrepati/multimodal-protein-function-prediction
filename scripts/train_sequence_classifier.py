"""Train the ProtT5 sequence classifier for EC-number prediction.

Workflow:
protein sequence -> ProtT5 encoder -> LoRA fine-tuning -> multiclass EC prediction

The original project adapted a public protein-language-model fine-tuning
workflow. This version retains the project-specific modeling approach while
removing unrelated benchmark and tutorial code.
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
from sklearn.model_selection import train_test_split
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
UNCOMMON_AMINO_ACIDS = re.compile(r"[OBUZJ]")


class ProtT5Classifier(nn.Module):
    """ProtT5 encoder with a mean-pooled classification head."""

    def __init__(
        self,
        encoder: T5EncoderModel,
        num_labels: int,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.encoder = encoder
        hidden_size = encoder.config.d_model

        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, num_labels)

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

        pooled = outputs.last_hidden_state.mean(dim=1)

        pooled = self.dropout(pooled)
        pooled = torch.tanh(self.dense(pooled))
        pooled = self.dropout(pooled)

        logits = self.out_proj(pooled)

        loss = None

        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )


def set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)


def prepare_sequences(sequences: pd.Series) -> list[str]:
    """Apply the amino-acid preprocessing used for ProtT5."""

    cleaned = sequences.astype(str).map(
        lambda sequence: UNCOMMON_AMINO_ACIDS.sub(
            "X",
            sequence.strip().upper(),
        )
    )

    return [
        " ".join(sequence)
        for sequence in cleaned
    ]


def tokenize_dataset(
    tokenizer: T5Tokenizer,
    sequences: list[str],
    labels: list[int],
) -> Dataset:
    """Tokenize protein sequences and attach labels."""

    tokenized = tokenizer(
        sequences,
        max_length=1024,
        padding=True,
        truncation=True,
    )

    dataset = Dataset.from_dict(tokenized)

    return dataset.add_column(
        "labels",
        labels,
    )


def build_model(
    num_labels: int,
) -> tuple[nn.Module, T5Tokenizer]:
    """Load ProtT5 and apply LoRA fine-tuning."""

    tokenizer = T5Tokenizer.from_pretrained(
        CHECKPOINT,
        do_lower_case=False,
    )

    encoder = T5EncoderModel.from_pretrained(
        CHECKPOINT
    )

    model = ProtT5Classifier(
        encoder=encoder,
        num_labels=num_labels,
    )

    lora_config = LoraConfig(
        r=4,
        lora_alpha=1,
        bias="all",
        target_modules=["q", "k", "v", "o"],
    )

    model = inject_adapter_in_model(
        lora_config,
        model,
    )

    # Train the classification head together with LoRA parameters.
    for parameter in model.dense.parameters():
        parameter.requires_grad = True

    for parameter in model.out_proj.parameters():
        parameter.requires_grad = True

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,} / {total_parameters:,}"
    )

    return model, tokenizer


def save_trainable_parameters(
    model: nn.Module,
    output_path: Path,
) -> None:
    """Save only parameters updated during fine-tuning."""

    trainable_state = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    torch.save(
        trainable_state,
        output_path,
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
            "UniProtID, Sequence, and EC columns."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/sequence"),
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

    set_seeds(args.seed)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = pd.read_csv(args.data)

    required_columns = {
        "UniProtID",
        "Sequence",
        "EC",
    }

    missing_columns = required_columns.difference(
        data.columns
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    data = data.dropna(
        subset=[
            "UniProtID",
            "Sequence",
            "EC",
        ]
    ).copy()

    label_encoder = LabelEncoder()

    data["label"] = label_encoder.fit_transform(
        data["EC"].astype(str)
    )

    # Reproduce the original 70 / 10 / 20 split.
    train_df, test_df = train_test_split(
        data,
        test_size=0.20,
        random_state=0,
    )

    train_df, val_df = train_test_split(
        train_df,
        test_size=0.125,
        random_state=7,
    )

    print(f"Train: {len(train_df)}")
    print(f"Validation: {len(val_df)}")
    print(f"Test: {len(test_df)}")

    print(
        f"EC classes: "
        f"{len(label_encoder.classes_)}"
    )

    model, tokenizer = build_model(
        num_labels=len(label_encoder.classes_)
    )

    train_dataset = tokenize_dataset(
        tokenizer,
        prepare_sequences(train_df["Sequence"]),
        train_df["label"].tolist(),
    )

    val_dataset = tokenize_dataset(
        tokenizer,
        prepare_sequences(val_df["Sequence"]),
        val_df["label"].tolist(),
    )

    test_dataset = tokenize_dataset(
        tokenizer,
        prepare_sequences(test_df["Sequence"]),
        test_df["label"].tolist(),
    )

    def compute_metrics(eval_prediction):

        logits, labels = eval_prediction

        predictions = np.argmax(
            logits,
            axis=1,
        )

        return {
            "accuracy": accuracy_score(
                labels,
                predictions,
            )
        }

    training_args = TrainingArguments(
        output_dir=str(
            args.output_dir / "trainer"
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Evaluate only on the held-out test set.
    test_output = trainer.predict(
        test_dataset
    )

    test_predictions = np.argmax(
        test_output.predictions,
        axis=1,
    )

    test_accuracy = accuracy_score(
        test_df["label"],
        test_predictions,
    )

    predicted_ec = label_encoder.inverse_transform(
        test_predictions
    )

    predictions_df = pd.DataFrame(
        {
            "UniProtID": test_df[
                "UniProtID"
            ].to_numpy(),
            "True_EC": test_df[
                "EC"
            ].to_numpy(),
            "Predicted_EC": predicted_ec,
        }
    )

    predictions_df.to_csv(
        args.output_dir
        / "test_predictions.csv",
        index=False,
    )

    metrics = {
        "checkpoint": CHECKPOINT,
        "train_records": len(train_df),
        "validation_records": len(val_df),
        "test_records": len(test_df),
        "num_ec_classes": len(
            label_encoder.classes_
        ),
        "test_accuracy": float(
            test_accuracy
        ),
        "epochs": args.epochs,
        "seed": args.seed,
    }

    (
        args.output_dir / "metrics.json"
    ).write_text(
        json.dumps(
            metrics,
            indent=2,
        ),
        encoding="utf-8",
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

    save_trainable_parameters(
        model,
        args.output_dir
        / "prot_t5_lora_trainable_parameters.pt",
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
