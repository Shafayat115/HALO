# HALO: Hybrid Attention Model for Subcellular Localization

We propose **HALO (Hybrid Attention Model for Subcellular
LOcalization)**, a novel framework that integrates structural information from the AlphaFold structure database and
semantic embeddings from large-scale protein language models, such as fine-tuned ESM (Evolutionary Scale Modeling).
The hybrid architecture uses graph attention networks (GATs) to incorporate biochemical, structural, and sequence-
derived features into a unified representation. Our model leverages ESM embeddings to capture evolutionary and
contextual knowledge of amino acid sequences, complemented by spatial information from AlphaFold-predicted structures.
In addition, it integrates amino acid biochemical properties, such as polarity and hydrophobicity, to enhance feature
diversity. A learnable weighted mechanism dynamically balances contributions from these distinct feature modalities.
We evaluate HALO on three datasets with minimal homology between the training and test sets, where the model
achieves state-of-the-art performance across key metrics. The hybrid approach effectively predicts subcellular localization
for proteins with and without structural data, bridging the gap in datasets where structural information is unavailable. In
particular, the model incorporates a focal loss function with a learnable threshold to address label imbalance and enhance
generalization. We showcase the potential of combining fine-tuned large-language models and AlphaFold structural
embeddings in graph-based neural networks, setting a new standard in subcellular localization prediction. HALO offers a
flexible, adaptable framework for broader protein function annotation tasks, emphasizing the importance of multi-modal
data integration in bioinformatics.

### Figures

<div align="center">
    <img src="https://github.com/user-attachments/assets/94573baa-bfeb-4a2f-b15b-91407b12b489" alt="Architecture" width="80%">
    <p><strong>Figure 1:</strong> Architecture Diagram</p>
</div>

<div align="center">
    <img src="https://github.com/user-attachments/assets/dfc8c5fc-485d-4a11-8bd7-763d0a192d6a" alt="Fine-Tune-ESM" width="80%">
    <p><strong>Figure 2:</strong> Fine-tuning ESM-2 Architecture Diagram</p>
</div>

<div align="center">
    <img src="https://github.com/user-attachments/assets/154a4015-432c-428f-ae0e-25ae9a676e60" alt="GAT-Model" width="80%">
    <p><strong>Figure 3:</strong> Graph Attention Model Training Diagram</p>
</div>



## Pre-requisites

Ensure you have the following installed:
- **Python**: Version 3.8 or higher
- **CUDA-compatible GPU**: With appropriate drivers installed
- **Conda**: For managing the Python environment
- **Git**: To clone repositories if needed

---

## Setup Instructions

### 1. Create and Activate Conda Environment
Run the following commands to create and activate a new Conda environment:
```bash
conda create --name halo_env python=3.8 -y
conda activate halo_env

---

## Install Dependencies
Ensure you have the requirements.txt file in your working directory. Install the required dependencies by running:
