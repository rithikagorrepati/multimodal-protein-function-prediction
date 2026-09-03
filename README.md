# Multimodal Protein Function Prediction

Multimodal deep learning for predicting protein enzymatic function from sequence, functional annotation, and predicted 3D structure.

This project compares three protein representations — **amino-acid sequence, UniProt functional annotation, and AlphaFold-predicted structure** — and combines their predictions in a multimodal ensemble for **Enzyme Commission (EC) number classification**.

---

## Overview

Protein function can be represented through several complementary sources of information:

- an amino-acid **sequence** contains biochemical and evolutionary information,
- a protein's **3D structure** captures spatial relationships relevant to molecular function,
- and a functional **annotation** contains biomedical and biological context.

The central question of this project was:

> **Can combining sequence, structure, and annotation information improve protein-function prediction compared with using any single modality alone?**

The prediction target was the protein's **Enzyme Commission (EC) number**, making this a large multiclass classification problem.

---

## Dataset

The project used human protein records assembled primarily from **UniProt** and **AlphaFold**.

| Dataset characteristic | Size |
| --- | ---: |
| Master protein records | 3,630 |
| Unique UniProt IDs | 3,620 |
| Distinct EC classes | 1,344 |
| Usable protein sequences | 3,611 |
| Processed protein structures | ~3,550 |

The unusually high number of EC classes relative to the number of proteins makes this a difficult classification problem. Many EC classes contain only a small number of examples.

Three representations of the proteins were used:

### Sequence

Amino-acid sequences were obtained from UniProt in FASTA format.

```text
UniProt FASTA
      ↓
   ProtT5
      ↓
 LoRA fine-tuning
      ↓
 EC prediction
```

### Functional Annotation

Protein descriptions and functional annotations from UniProt were represented using **BioBERT**.

During the original Georgia Tech group project, **Rithika Gorrepati developed and evaluated BioBERT-based MLP/CNN approaches for EC prediction**. The final team workflow used for the multimodal comparison employed a **BioBERT-based RNN classifier**, which is the annotation architecture represented in the reorganized pipeline in this repository.

```text
UniProt annotation
        ↓
      BioBERT
        ↓
sentence embedding
        ↓
   RNN classifier
        ↓
    EC prediction
```
### Structure

AlphaFold-predicted PDB structures were processed through ProteinMPNN to obtain numerical structural representations.

```text
AlphaFold PDB
      ↓
 ProteinMPNN
      ↓
structural embedding
      ↓
 Transformer
      ↓
 EC prediction
```

---

## Multimodal Architecture

The project trained one model for each modality and then combined their class-level outputs using a multi-head-attention ensemble.

```text
                  Protein Sequence
                        │
                        ▼
                  ProtT5 + LoRA
                        │
                        │
                        ▼
                 Sequence scores
                        │
                        │
                        ├───────────────┐
                        │               │
UniProt Annotation     │               │       AlphaFold Structure
        │               │               │               │
        ▼               │               │               ▼
     BioBERT            │               │          ProteinMPNN
        │               │               │               │
        ▼               │               │               ▼
Annotation Classifier   │               │       Structure Transformer
        │               │               │               │
        ▼               │               │               ▼
 Annotation scores ─────┤               ├────── Structure scores
                        │               │
                        └───────┬───────┘
                                ▼
                     Multi-Head Attention
                                │
                                ▼
                         EC Classification
```

The four final model families were therefore:

1. **Sequence model** — ProtT5 with LoRA fine-tuning
2. **Annotation model** — BioBERT-based annotation classification
3. **Structure model** — ProteinMPNN representations with a Transformer classifier
4. **Multimodal model** — multi-head attention over the three base-model outputs

---

## Original Project Results

The original course experiments found that the **sequence model was the strongest individual modality**, while the structural model performed substantially worse.

| Model | Approx. validation accuracy |
| --- | ---: |
| Sequence | ~55% |
| Annotation | ~55% |
| Structure | ~27% |
| Multimodal ensemble | ~60% |

On the original held-out test set:

- **Sequence model:** ~54% accuracy
- **Multimodal ensemble:** ~55% accuracy

The ensemble therefore produced only a **small improvement over the strongest single-modality model**.

That was an important result rather than simply a modeling failure: adding additional biological modalities and model complexity did **not automatically produce a large improvement**.

The original report also recorded an ensemble AUROC of approximately **0.93**, although accuracy is used here as the primary headline metric because of the extreme sparsity of the multiclass problem.

---

## Key Findings

### Sequence information was the strongest single modality

ProtT5 performed substantially better than the structure-only model and remained very close to the final multimodal ensemble on the test set.

### Multimodal learning provided only a modest gain

Combining all three modalities increased test accuracy from approximately **54% to 55%**.

This suggested that the additional computational complexity of the ensemble was not clearly justified by the improvement observed in this dataset.

### Structural representation was the main bottleneck

The original structural workflow lost information while transforming variable-length ProteinMPNN representations into a form suitable for classification.

The structure-only classifier achieved less than 30% validation accuracy and was the weakest branch of the system.

### Class sparsity was a major limitation

The dataset contained roughly:

```text
3,600 proteins
       ↓
1,300+ EC classes
```

Some EC classes had very few examples, including classes represented by only a single protein. This makes reliable learning and evaluation difficult, particularly when an EC class is absent from the training partition.

---

## Repository Structure

```text
multimodal-protein-function-prediction/
│
├── docs/
│   └── data_description.md
│
├── scripts/
│   ├── download_uniprot_sequences.py
│   ├── build_sequence_dataset.py
│   ├── create_split_manifest.py
│   ├── train_sequence_classifier.py
│   ├── extract_biobert_embeddings.py
│   ├── train_annotation_classifier.py
│   ├── extract_proteinmpnn_embeddings.py
│   ├── train_structure_classifier.py
│   └── train_multimodal_ensemble.py
│
├── .gitignore
└── README.md
```

---

## Repository Implementation

The original work was completed as a Georgia Tech group project using Python, PyTorch, Jupyter notebooks, and modality-specific deep-learning workflows.

This repository reorganizes the project into modular scripts for data preparation, representation learning, model training, and multimodal integration. The cleaned structure improves reproducibility and makes the sequence, annotation, structure, and ensemble workflows easier to understand and run independently.

---

## Shared Data Splitting

One issue identified during reconstruction was that the original modality notebooks did not always use identical protein-level train/validation/test assignments.

The cleaned workflow therefore introduces a shared:

```text
split_manifest.csv
```

that assigns each UniProt accession to one of:

```text
train
validation
test
```

All modalities can then reference the same protein-level partition.

This helps prevent a protein from being treated as training data in one modality while simultaneously appearing in the test set of another modality.

---

## Leakage-Aware Multimodal Evaluation

Another important consideration is stacked-model evaluation.

If the ensemble is trained using predictions from a base model on samples that the base model itself saw during training, the ensemble can receive overly optimistic features.

A strict multimodal stacking workflow should therefore use:

- **out-of-fold predictions for ensemble training**, and
- predictions from untouched validation/test proteins for final evaluation.

The cleaned repository is being structured around this more rigorous evaluation framework rather than reproducing leakage-prone behavior from the original experimental notebooks.

---

## Tools and Models

### Programming

- Python
- NumPy
- pandas

### Machine Learning / Deep Learning

- PyTorch
- scikit-learn
- Hugging Face Transformers
- PEFT / LoRA

### Biological Representation Models

- **ProtT5-XL-UniRef50** — protein sequence representation and fine-tuning
- **BioBERT** — biomedical text representation
- **ProteinMPNN** — structural representation from protein coordinates

### Biological Data Sources

- UniProt
- AlphaFold Protein Structure Database

---

## External Dependencies

### ProteinMPNN

ProteinMPNN is an external research model and is **not vendored into this repository**.

The structural embedding script expects a separate ProteinMPNN installation and pretrained checkpoint.

This repository uses ProteinMPNN as a dependency rather than presenting its source code as part of this project.

### Raw Data

Large raw biological files are not stored in this repository.

These include:

- downloaded UniProt FASTA files,
- AlphaFold PDB structures,
- generated ProteinMPNN embeddings,
- model checkpoints,
- and other intermediate model artifacts.

The repository instead provides code describing how those data were prepared and processed.

---

## Project Context

This project originated as a **Georgia Tech group project** completed as part of a machine-learning course.

**Team:** Joshua Joseph, Rithika Gorrepati, Joshua Traynelis, and Vishank Raghavan.

The original project was collaborative, and the complete multimodal workflow should not be interpreted as the work of a single team member.

The project report specifically documents **Rithika Gorrepati's contribution** as developing and evaluating **BioBERT-based MLP/CNN approaches for EC prediction** and contributing to the **Conclusion and Future Work** analysis.

The final team project additionally included the ProtT5 sequence model, RNN annotation classifier, ProteinMPNN structural workflow, structural Transformer, and multimodal attention ensemble.

This repository reorganizes the collaborative Georgia Tech course project into a cleaner portfolio and reproducibility-oriented implementation.

---

## Limitations

Several limitations should be considered when interpreting the original results:

- More than 1,300 EC classes were represented by only ~3,600 proteins.
- Many EC classes had very few training examples.
- Not every protein had usable data for every modality.
- The structural representation pipeline lost information during dimensional standardization in the original implementation.
- Multimodal integration improved performance only slightly over ProtT5 alone.
- Original modality-specific notebooks did not always use identical data splits.
- The original ensemble workflow requires stricter out-of-fold evaluation to establish a leakage-controlled stacking benchmark.

---

## What I Learned

The most useful result from this project was not simply that an ensemble produced the highest accuracy.

It showed that **more modalities do not necessarily mean a better model**.

Sequence representations captured most of the predictive signal available in this dataset, while structural modeling remained limited by the representation and architecture used. The project also highlighted how data sparsity, alignment across modalities, train/test splitting, and ensemble design can matter as much as model complexity in biological machine learning.

---

## Future Work

Potential improvements include:

- collecting more examples for underrepresented EC classes,
- hierarchical EC-number prediction rather than treating every complete EC number as an unrelated class,
- class-aware or grouped evaluation strategies,
- improved structural representations that preserve residue and spatial information,
- graph neural networks for structural modeling,
- simpler ensemble methods as baselines,
- calibrated confidence estimates,
- out-of-fold base-model predictions for leakage-controlled stacking,
- and systematic ablation studies to quantify the contribution of each modality.

---

## References

This project builds on publicly available biological resources and pretrained models including:

- UniProt
- AlphaFold
- ProteinMPNN
- ProtT5
- BioBERT
