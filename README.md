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
```

## 1. Install Dependencies

Ensure you have the `requirements.txt` file in your working directory. Install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

For PyTorch Geometric dependencies, install them separately according to your PyTorch version and CUDA setup. For example, if using CUDA 11.8:

```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
```

## 2. Dataset Setup

Prepare the following files and directories in your working directory:

### Datasets:
Ensure the following CSV files are available:
- `deeploc_train.csv`: Training dataset
- `deeploc_validation.csv`: Validation dataset
- `deeploc_test.csv`: Test dataset

Each CSV file should include the following columns:
- **ACC**: Unique identifier for each sequence.
- **Sequence**: The protein sequence.
- **Label columns**: Binary values (`0` or `1`) for subcellular locations such as `Cytoplasm`, `Nucleus`, `Extracellular`, etc.

---

### Additional Directories:

1. **PDB Files Directory**:
   - Create a directory named `pdb_files`.
   - Place AlphaFold-predicted PDB files for the sequences in your dataset.
   - If any PDB files are missing, the scripts will automatically attempt to download them.

2. **ESM Embeddings Directory**:
   - Create a directory named `esm_embeddings_pretrained`.
   - This directory will store the generated ESM embeddings.

## Running the Scripts

Follow the steps below to execute the scripts in the correct order.

---

### Step 1: Fine-Tune the ESM Model

Fine-tune the ESM model and generate embeddings by running the `Fine-Tune-ESM-Model.py` script.

```bash
python Fine-Tune-ESM-Model.py
```
### Expected Outputs:

- Fine-tuned model weights saved in the `Models` directory.
- Logs written to `fine_tuned_esm_with_structure_weight_fixed_weight_combo.log`.

---

### Step 2: Train Weighted ESM-GAT Hybrid Model

Train a weighted ESM-GAT hybrid model by running the `Fine-Tuned_Weighted_ESM_GAT_Fixed_Combo.py` script.

```bash
python Fine-Tuned_Weighted_ESM_GAT_Fixed_Combo.py
```
### Expected Outputs:

- Logs saved in `fine_tuned_esm_with_structure_weight_fixed_weight_combo.log`.
- Best model weights saved in the `Models` directory.
- Evaluation metrics and confidence analysis saved as:
  - `confidence_analysis_test_set_updated.json`
  - `evaluation_results_updated.csv`

---

### Step 3: Run the HALO Framework

To combine ESM and GAT predictions and improve uncertain predictions, run the following command:

```bash
python HALO.py
```
### Expected Outputs:

- Predictions for uncertain sequences saved in `uncertain_sequences_with_gat.csv`.
- Final evaluation metrics saved as:
  - `esm_results.json`
  - `final_test_metrics_updated.json`
