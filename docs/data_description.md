# Data Description

This project used multimodal human protein data to predict Enzyme Commission (EC) classes.

## Master Dataset

The project dataset contained:

- 3,630 protein records
- 3,620 unique UniProt IDs
- 1,344 distinct EC classes

The master table included the following fields:

- Gene
- Chromosomal position
- UniProt ID
- Protein description / functional annotation
- Enzyme Commission (EC) number

## Data Modalities

Three complementary representations were collected for the proteins.

### Protein Sequence

Amino-acid sequences were obtained from UniProt in FASTA format and used for sequence-based EC classification with ProtT5.

The sequence-classification dataset contained 3,611 usable protein sequences.

### Protein Annotation

Functional protein descriptions from UniProt were used as the textual modality.

The annotations were converted into numerical representations using BioBERT embeddings and used for annotation-based EC classification.

### Protein Structure

AlphaFold-predicted protein structures in PDB format were used as the structural modality.

Approximately 3,550 protein structures were processed through ProteinMPNN to generate structural representations for downstream classification.

## Prediction Target

The target variable was the protein's Enzyme Commission number.

Because the dataset contained approximately 3,600 proteins distributed across more than 1,300 EC classes, the project involved a highly sparse multiclass classification problem.

## Data Availability

Raw protein sequences and AlphaFold structure files are not included in this repository.

The project used data derived from publicly available biological resources, primarily UniProt and AlphaFold.
