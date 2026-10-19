# Transformer Architecture Analysis for Sentiment Classification

## Overview

This project investigates how Transformer attention-head count and encoder depth affect model performance and attention structure in a binary sentiment-classification task.

Rather than relying on a pretrained Transformer, I implemented a compact encoder-only Transformer in PyTorch to examine its internal components directly. Four architectural configurations are compared, using attention visualisation and spectral analysis to investigate representational similarities and potential redundancy between attention heads.

## Objectives

- Implement a Transformer encoder and classification head from core components.
- Compare single-head and multi-head attention, alongside different encoder depths.
- Evaluate training and validation performance across multiple random seeds.
- Investigate attention-head behaviour using visualisation, cosine similarity, singular value decomposition (SVD) and numerical rank.

## Implementation

The model is based on the Transformer architecture introduced by Vaswani et al. (2017), using *The Annotated Transformer* as a reference.

Core components implemented in PyTorch include:

- Token embeddings and sinusoidal positional encoding
- Scaled dot-product attention
- Multi-head self-attention
- Position-wise feed-forward networks
- Residual connections and layer normalisation
- An encoder stack and sentiment-classification head

The model uses a hidden dimension (`d_model`) of 32 and a feed-forward dimension (`d_ff`) of 64. This compact configuration makes it practical to inspect attention matrices and compare architectural variations without the computational cost of a large pretrained model.

## Dataset

The experiments use 10,000 movie reviews from the IMDB sentiment dataset. Each review is labelled as positive or negative.

The IMDB dataset was selected because its relatively long reviews provide opportunities to investigate attention over longer text sequences.

The implementation uses the BERT uncased tokenizer for tokenisation; the BERT model itself is not used.

## Experiments

Four model configurations were compared:

| Configuration | Attention heads | Encoder layers |
|---|---:|---:|
| Single-head Transformer | 1 | 1 |
| Multi-head Transformer | 4 | 1 |
| Multi-head Transformer | 8 | 1 |
| Deeper multi-head Transformer | 4 | 2 |

Each configuration was trained across three random seeds and evaluated over eight epochs. Training and validation loss and accuracy were recorded to compare performance, convergence and overfitting.

## Results

The following figures show the mean across three seeds of each run's maximum validation accuracy:

| Configuration | Maximum validation accuracy |
|---|---:|
| Single-head Transformer | 85.1% |
| 4 heads, 1 layer | 85.4% |
| 8 heads, 1 layer | 84.6% |
| 4 heads, 2 layers | 82.8% |

The four-head, single-layer configuration achieved the highest mean of the per-seed peak validation accuracies, although its advantage over the single-head model was small.

Increasing the number of attention heads or encoder depth did not improve performance in these experiments. The larger configurations also showed evidence of overfitting.

These results reflect the experimental setup described in the accompanying report and are not intended as a benchmark against modern pretrained sentiment-classification models.

## Attention Analysis

Beyond classification accuracy, the project investigates the structure of the learned attention matrices.

### Attention visualisation

Attention matrices were visualised for selected reviews to examine patterns including local positional attention, relationships between tokens and attention to sentiment-bearing words.

Examples showed that different heads could focus on overlapping patterns, while in other cases multi-head attention distributed attention across broader contextual relationships.

### Cosine similarity

Pairwise cosine similarity was used to compare attention heads. For one example, the cosine similarity between the single-head attention matrix and the mean attention matrix of the four-head model was approximately **0.89**, indicating similar attention patterns for that input.

### Singular value decomposition and effective rank

SVD and numerical rank were used to investigate the effective dimensionality of attention matrices.

The analysis suggested that matrices from configurations with more heads often had lower individual effective ranks, while different examples exhibited different patterns.

These findings are exploratory: attention patterns depend on the input, and numerical rank depends on the chosen tolerance. A broader analysis across the full dataset would be needed to establish more general conclusions.

## Key Takeaways

- Increasing attention-head count does not necessarily improve performance on a given task.
- Some heads appear to capture overlapping attention patterns, while others show different patterns on selected examples.
- Classification accuracy alone does not explain how an attention mechanism behaves internally.
- Attention visualisation and spectral analysis provide complementary ways to investigate Transformer representations.
- Model depth, head count, regularisation and dataset complexity are useful dimensions for further experimentation.

## Limitations and Future Work

- Expand the analysis across the complete evaluation dataset rather than selected examples.
- Improve regularisation and training configuration to investigate overfitting.
- Use a dedicated validation split from the training data and reserve the official test set for final evaluation.
- Apply attention masks consistently so padding tokens do not contribute to attention or pooled representations.
- Investigate more challenging NLP tasks and evaluate whether observed attention patterns generalise across inputs and model configurations.

## Technologies

- Python
- PyTorch
- Hugging Face Tokenizers
- NumPy
- Matplotlib
- Singular Value Decomposition
- Cosine Similarity

## References

- Vaswani et al. (2017). *Attention Is All You Need.*
- Rush et al. (2022). *The Annotated Transformer.*
- Maas et al. (2011). *Learning Word Vectors for Sentiment Analysis.*
