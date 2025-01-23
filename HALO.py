import torch
from torch.utils.data import DataLoader
from transformers import EsmTokenizer, EsmForSequenceClassification
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.data import Data, Dataset
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import logging
from transformers import EsmTokenizer, EsmForSequenceClassification
import os
import json
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score, jaccard_score, matthews_corrcoef
from Bio.PDB import PDBParser
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Dataset
import requests

# Logging setup
logging.basicConfig(filename="evaluation_combined.log", level=logging.INFO, format="%(asctime)s - %(message)s")

# Device configuration
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Label columns
label_columns = ["Cytoplasm", "Nucleus", "Extracellular", "Cell membrane",
                 "Mitochondrion", "Plastid", "Endoplasmic reticulum",
                 "Lysosome/Vacuole", "Golgi apparatus", "Peroxisome"]

# Thresholds
esm_threshold_low = 0.40
esm_threshold_high = 0.50
gat_threshold = 0.40

# Load tokenizer
tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")


pdb_dir = "pdb_files"
esm_dir = "esm_embeddings_pretrained"
fine_tuned_esm_path = "Models/fine_tuned_esm_attention_model_epoch_10.pth"
esm_model = EsmForSequenceClassification.from_pretrained(
    "facebook/esm2_t33_650M_UR50D",
    num_labels=len(label_columns),
    output_hidden_states=True
).to(device)
esm_tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")



# Amino acid biochemical properties
AMINO_ACIDS = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
            "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
            "THR", "TRP", "TYR", "VAL"]

AMINO_ACID_PROPERTIES = {
    "ALA": {"hydrophobicity": 1.8, "charge": 0, "polarity": 8.1, "weight": 89.1},
    "ARG": {"hydrophobicity": -4.5, "charge": 1, "polarity": 10.5, "weight": 174.2},
    "ASN": {"hydrophobicity": -3.5, "charge": 0, "polarity": 11.6, "weight": 132.1},
    "ASP": {"hydrophobicity": -3.5, "charge": -1, "polarity": 13.0, "weight": 133.1},
    "CYS": {"hydrophobicity": 2.5, "charge": 0, "polarity": 5.5, "weight": 121.2},
    "GLN": {"hydrophobicity": -3.5, "charge": 0, "polarity": 10.5, "weight": 146.2},
    "GLU": {"hydrophobicity": -3.5, "charge": -1, "polarity": 12.3, "weight": 147.1},
    "GLY": {"hydrophobicity": -0.4, "charge": 0, "polarity": 9.0, "weight": 75.1},
    "HIS": {"hydrophobicity": -3.2, "charge": 0.5, "polarity": 10.4, "weight": 155.2},
    "ILE": {"hydrophobicity": 4.5, "charge": 0, "polarity": 5.2, "weight": 131.2},
    "LEU": {"hydrophobicity": 3.8, "charge": 0, "polarity": 4.9, "weight": 131.2},
    "LYS": {"hydrophobicity": -3.9, "charge": 1, "polarity": 11.3, "weight": 146.2},
    "MET": {"hydrophobicity": 1.9, "charge": 0, "polarity": 5.7, "weight": 149.2},
    "PHE": {"hydrophobicity": 2.8, "charge": 0, "polarity": 5.2, "weight": 165.2},
    "PRO": {"hydrophobicity": -1.6, "charge": 0, "polarity": 8.0, "weight": 115.1},
    "SER": {"hydrophobicity": -0.8, "charge": 0, "polarity": 9.2, "weight": 105.1},
    "THR": {"hydrophobicity": -0.7, "charge": 0, "polarity": 8.6, "weight": 119.1},
    "TRP": {"hydrophobicity": -0.9, "charge": 0, "polarity": 5.4, "weight": 204.2},
    "TYR": {"hydrophobicity": -1.3, "charge": 0, "polarity": 6.2, "weight": 181.2},
    "VAL": {"hydrophobicity": 4.2, "charge": 0, "polarity": 5.9, "weight": 117.1}
}


# Define the GAT model
class GATModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=4, heads=4):
        super(GATModel, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(GATConv(input_dim, hidden_dim, heads=heads, concat=True))
        self.bns.append(nn.BatchNorm1d(hidden_dim * heads))

        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim * heads))

        self.convs.append(GATConv(hidden_dim * heads, output_dim, heads=1, concat=False))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs[:-1], self.bns):
            x = F.elu(bn(conv(x, edge_index)))
            x = F.dropout(x, p=0.6, training=self.training)
        x = self.convs[-1](x, edge_index)
        pooled = global_mean_pool(x, batch)
        return pooled

# Define ESM with attention
class AttentionHead(nn.Module):
    def __init__(self, embedding_dim):
        super(AttentionHead, self).__init__()
        self.attention_weights = nn.Linear(embedding_dim, 1)

    def forward(self, embeddings):
        scores = self.attention_weights(embeddings)
        scores = F.softmax(scores, dim=1)
        weighted_embeddings = embeddings * scores
        aggregated_embedding = torch.sum(weighted_embeddings, dim=1)
        return aggregated_embedding

class ESMWithAttention(nn.Module):
    def __init__(self, base_model, embedding_dim, num_labels):
        super(ESMWithAttention, self).__init__()
        self.base_model = base_model
        self.attention_head = AttentionHead(embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_labels)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        esm_embeddings = outputs.hidden_states[-1]
        attention_embedding = self.attention_head(esm_embeddings)
        logits = self.classifier(attention_embedding)
        return logits

# Dataset for ESM
class ProteinSequenceDatasetCSV(Dataset):
    def __init__(self, csv_file, label_columns, tokenizer, max_length=1024):
        self.data = pd.read_csv(csv_file)
        self.sequences = self.data["Sequence"].tolist()
        self.labels = self.data[label_columns].values
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        labels = self.labels[idx].astype(float)
        tokens = self.tokenizer(sequence, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt")
        return tokens["input_ids"].squeeze(), tokens["attention_mask"].squeeze(), torch.tensor(labels, dtype=torch.float32)


def get_biochemical_properties(residue_name):
    if residue_name in AMINO_ACID_PROPERTIES:
        return list(AMINO_ACID_PROPERTIES[residue_name].values())
    return [0.0, 0, 0.0, 0.0]




# Dataset for GAT
class PDBGraphDataset(Dataset):
    def __init__(self, pdb_files, labels, esm_dir, weights, edge_threshold=10.0):
        super().__init__()
        self.pdb_files = pdb_files
        self.labels = labels
        self.esm_dir = esm_dir
        self.weights = weights
        self.edge_threshold = edge_threshold
        self.parser = PDBParser(QUIET=True)

    def len(self):
        return len(self.pdb_files)

    def get(self, idx):
        pdb_file = self.pdb_files[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        # Ensure label is of shape [1, 10] instead of [10]
        if label.ndim == 1:
            label = label.unsqueeze(0)

        # Load the ESM embedding
        esm_embedding_file = os.path.join(
            self.esm_dir, f"{os.path.basename(pdb_file).split('.')[0]}_esm.npy"
        )
        esm_embedding = torch.tensor(np.load(esm_embedding_file), dtype=torch.float).to(device)

        if os.path.exists(pdb_file):
            structure = self.parser.get_structure("protein", pdb_file)
            residues = list(structure.get_residues())

            nodes, edges, edge_weights = [], [], []
            embedding_len = esm_embedding.shape[0]  # Number of residue-level embeddings
            embedding_dim = esm_embedding.shape[1]  # Dimensionality of each embedding

            for i, residue in enumerate(residues):
                if 'CA' not in residue:
                    continue

                ca_atom = residue['CA']
                amino_acid = residue.get_resname()

                # 3D coordinates
                coord_features = [c * self.weights["coord_weight"] for c in ca_atom.coord]
                
                # ESM embedding: If i >= embedding_len, use a zero vector
                if i < embedding_len:
                    esm_features = (esm_embedding[i] * self.weights["esm_weight"]).tolist()
                else:
                    esm_features = [0.0] * embedding_dim  # Zero vector fallback

                # Biochemical properties
                bio_features = [
                    p * self.weights["bio_weight"]
                    for p in get_biochemical_properties(amino_acid)
                ]

                # Combine all features
                node_features = coord_features + esm_features + bio_features
                nodes.append(node_features)

            # Build edges based on distance threshold
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    dist = np.linalg.norm(
                        np.array(nodes[i][:3]) - np.array(nodes[j][:3])
                    )
                    if dist < self.edge_threshold:
                        edges.append((i, j))
                        edges.append((j, i))
                        # Weighted edges
                        edge_weights.append(1.0 / dist)
                        edge_weights.append(1.0 / dist)

            # Construct the data object
            x = torch.tensor(nodes, dtype=torch.float)
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_weights = torch.tensor(edge_weights, dtype=torch.float)
        else:
            # If PDB is missing, just use ESM embeddings as before
            x = esm_embedding * self.weights["esm_weight"]
            edge_index = torch.empty((2, 0), dtype=torch.long)  # No edges
            edge_weights = torch.empty(0, dtype=torch.float)    # No edge attributes

        return Data(x=x, edge_index=edge_index, edge_attr=edge_weights, y=label)
    
    
def save_esm_embeddings(input_csv, esm_output_dir, batch_size=8, max_length=1024):
    df = pd.read_csv(input_csv)
    os.makedirs(esm_output_dir, exist_ok=True)
    
    sequences = df['Sequence'].tolist()
    acc_ids = df['ACC'].tolist()
    logging.info(f"Generating ESM embeddings for {input_csv}...")
    for i in range(0, len(sequences), batch_size):
        batch_sequences = sequences[i:i + batch_size]
        batch_acc_ids = acc_ids[i:i + batch_size]
        
        tokens = esm_tokenizer(batch_sequences, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            outputs = esm_model(**tokens)
            hidden_states = outputs.hidden_states  # Access all hidden layers
            if hidden_states is None:
                raise ValueError("Hidden states are not available. Ensure 'output_hidden_states=True' when initializing the model.")
            embeddings = hidden_states[-1].cpu().numpy()  # Use the last hidden layer
        
        for j, acc_id in enumerate(batch_acc_ids):
            output_file = os.path.join(esm_output_dir, f"{acc_id}_esm.npy")
            if not os.path.exists(output_file):
                np.save(output_file, embeddings[j])
                logging.info(f"Saved ESM embedding for {acc_id}.")
            else:
                logging.info(f"ESM embedding for {acc_id} already exists.")
    logging.info(f"Finished generating ESM embeddings for {input_csv}.")



def download_pdb(uniprot_id):
    logging.info(f"Attempting to download PDB file for {uniprot_id}.")

    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise HTTPError for bad responses
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {uniprot_id}: {e}")
        with open("missing_pdb_files.log", "a") as log_file:
            log_file.write(f"{uniprot_id}: {e}\n")
        return None
    if response.status_code == 200:
        pdb_path = os.path.join(pdb_dir, f"{uniprot_id}.pdb")
        with open(pdb_path, "wb") as f:
            f.write(response.content)
        print(f"Downloaded {uniprot_id}.pdb")
        logging.info(f"Successfully downloaded PDB file for {uniprot_id}.")

        return pdb_path
    else:
        print(f"Failed to download {uniprot_id}")
        with open("missing_pdb_files.log", "a") as log_file:
            log_file.write(f"{uniprot_id}\n")
        logging.warning(f"Failed to download PDB file for {uniprot_id}.")

        return None



# Load data
def load_data(csv_path):
    pdb_files, labels = [], []
    for _, row in pd.read_csv(csv_path).iterrows():
        acc_id = row['ACC']
        pdb_file = os.path.join(pdb_dir, f"{acc_id}.pdb")
        
        # Check if the PDB file exists or attempt download
        if not os.path.exists(pdb_file):
            pdb_file = download_pdb(acc_id)
            if not pdb_file:
                logging.warning(f"Skipping {acc_id}: PDB file could not be downloaded.")
                continue

        # Check for valid labels
        if row[label_columns].isna().any():
            logging.warning(f"Skipping {acc_id}: Invalid/missing labels.")
            continue

        pdb_files.append(pdb_file)
        labels.append(row[label_columns].values.astype(float))
    
    return pdb_files, labels

def process_dataset(csv_path, esm_output_dir):
    #Check if embeddings already exist
    if not os.path.exists(esm_output_dir) or not os.listdir(esm_output_dir):
        #logging.info(f"No embeddings found in {esm_output_dir}. Generating embeddings...")
        save_esm_embeddings(csv_path, esm_output_dir)
    #Load data with PDB file downloads if necessary
    return load_data(csv_path)

def evaluate_esm(loader_esm, esm_model, device, esm_threshold_low, esm_threshold_high):
    esm_model.eval()
    esm_predictions = []
    esm_labels = []
    esm_probabilities = []
    uncertain_indices = []
    sequence_ids = loader_esm.dataset.data["ACC"].tolist()  # Assuming ACC column holds sequence IDs
    uncertain_rows = []

    with torch.no_grad():
        for idx, (input_ids, attention_mask, labels) in enumerate(loader_esm):
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            # Forward pass
            outputs = esm_model(input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(outputs)

            # Predictions and uncertainty detection
            preds = (probs > esm_threshold_high).float()
            uncertain = ((probs > esm_threshold_low) & (probs < esm_threshold_high)).any(dim=1)

            esm_predictions.extend(preds.cpu().numpy())
            esm_labels.extend(labels.cpu().numpy())
            esm_probabilities.extend(probs.cpu().numpy())

            # Track uncertain sequences
            for i, flag in enumerate(uncertain):
                if flag:
                    dataset_index = idx * loader_esm.batch_size + i
                    if dataset_index < len(sequence_ids):  # Ensure index is valid
                        uncertain_indices.append(dataset_index)

                        # Prepare row data for the CSV
                        acc = sequence_ids[dataset_index]
                        confidence = probs[i].cpu().numpy()
                        uncertain_labels = labels[i].cpu().numpy()
                        row = {
                            "ACC": acc,
                            "esm_confidence": ",".join(f"{conf:.2f}" for conf in confidence),
                            "Uncertain_labels": ",".join(
                                f"{label_columns[j]}:{uncertain_labels[j]:.1f}" 
                                for j in range(len(label_columns))
                            ),
                            "GAT_confidence": None,  # Placeholder for GAT confidence
                            "Removed/Agreed": None  # Placeholder
                        }
                        uncertain_rows.append(row)

    # Save to CSV
    uncertain_df = pd.DataFrame(uncertain_rows)
    uncertain_df.to_csv("uncertain_sequences.csv", index=False)

    logging.info(f"Number of uncertain sequences: {len(uncertain_indices)}")
    logging.info(f"Uncertain sequences saved to 'uncertain_sequences.csv'.")

    # Prepare results for return
    esm_results = {
        "predictions": [pred.tolist() for pred in esm_predictions],
        "labels": [label.tolist() for label in esm_labels],
        "probabilities": [prob.tolist() for prob in esm_probabilities],
        "uncertain_indices": uncertain_indices,
    }
    with open("esm_results.json", "w") as f:
        json.dump(esm_results, f)

    return esm_results, uncertain_df




def filter_for_gat(test_dataset_gat, uncertain_indices):
    max_index = len(test_dataset_gat) - 1
    valid_indices = [idx for idx in uncertain_indices if 0 <= idx <= max_index]
    invalid_indices = [idx for idx in uncertain_indices if idx not in valid_indices]

    if invalid_indices:
        logging.warning(f"Ignored {len(invalid_indices)} invalid indices: {invalid_indices}.")
    logging.info(f"Valid uncertain indices: {len(valid_indices)}")
    
    return torch.utils.data.Subset(test_dataset_gat, valid_indices)

def evaluate_gat(loader_gat, gat_model, device):
    gat_model.eval()
    gat_predictions = []
    gat_labels = []

    with torch.no_grad():
        for data in loader_gat:
            data = data.to(device)
            output = gat_model(data)
            preds = torch.sigmoid(output).cpu().numpy()
            labels = data.y.cpu().numpy()

            gat_predictions.extend(preds)
            gat_labels.extend(labels)

    return gat_predictions, gat_labels

def combine_results_with_validation(esm_results, gat_predictions, gat_labels, uncertain_df, uncertain_indices, gat_threshold):
    final_predictions = np.array(esm_results["predictions"])  # Original ESM predictions
    gat_predictions = np.array(gat_predictions)  # Continuous GAT predictions

    agreed_sequences = 0
    removed_labels = 0

    for i, idx in enumerate(uncertain_indices):
        if i >= len(gat_predictions):
            logging.warning(f"Index {i} exceeds the size of GAT predictions ({len(gat_predictions)}). Skipping this index.")
            continue  # Skip indices that are out of bounds

        # Validate uncertain predictions using GAT
        gat_pred_binary = (gat_predictions[i] > gat_threshold).astype(int)  # Binarize GAT predictions

        # Update CSV fields
        gat_confidence = ",".join(f"{conf:.2f}" for conf in gat_predictions[i])
        original_prediction = final_predictions[idx]
        removed = int((original_prediction != (original_prediction * gat_pred_binary)).sum())

        uncertain_df.loc[i, "GAT_confidence"] = gat_confidence
        uncertain_df.loc[i, "Removed/Agreed"] = "Agreed" if removed == 0 else "Removed"

        # Combine results
        final_predictions[idx] *= gat_pred_binary

    # Save updated CSV
    uncertain_df.to_csv("uncertain_sequences_with_gat.csv", index=False)
    logging.info(f"Final uncertain sequences saved to 'uncertain_sequences_with_gat.csv'.")

    return final_predictions



# Load models
esm_model = ESMWithAttention(
    base_model=EsmForSequenceClassification.from_pretrained("facebook/esm2_t33_650M_UR50D", num_labels=len(label_columns)).base_model,
    embedding_dim=1280,
    num_labels=len(label_columns)
).to(device)
esm_model.load_state_dict(torch.load("Models/fine_tuned_esm_attention_model_epoch_10.pth", map_location=device))

gat_model = GATModel(input_dim=1280 + 3 + 4, hidden_dim=128, output_dim=len(label_columns)).to(device)
gat_model.load_state_dict(torch.load("best_model_0.5_1.0_0.5.pth", map_location=device))

# Load datasets
test_csv_file = "deeploc_test.csv"
test_dataset_esm = ProteinSequenceDatasetCSV(test_csv_file, label_columns, tokenizer)
test_loader_esm = DataLoader(test_dataset_esm, batch_size=8, shuffle=False)

test_pdb_files, test_labels = process_dataset(test_csv_file, "esm_embeddings_pretrained")
test_dataset_gat = PDBGraphDataset(test_pdb_files, test_labels, "esm_embeddings_pretrained", {"coord_weight": 0.5, "esm_weight": 1.0, "bio_weight": 0.5})
test_loader_gat = PyGDataLoader(test_dataset_gat, batch_size=8, shuffle=False)


# Evaluate
# Step 1: Evaluate ESM model
esm_results, uncertain_df = evaluate_esm(test_loader_esm, esm_model, device, esm_threshold_low, esm_threshold_high)

logging.info("ESM model evaluation completed. Results saved.")
# Step 2: Filter data for GAT evaluation based on ESM uncertainty
esm_results_dict = esm_results  # Directly assign the dictionary part
uncertain_indices = esm_results_dict["uncertain_indices"]
logging.info(f"Uncertain indices to subset: {uncertain_indices}")

# Use the uncertain indices for GAT filtering
gat_uncertain_dataset = filter_for_gat(test_dataset_gat, uncertain_indices)

if len(gat_uncertain_dataset) == 0:
    logging.warning("No uncertain sequences for GAT evaluation.")
    gat_predictions, gat_labels = [], []
else:
    gat_loader_filtered = PyGDataLoader(gat_uncertain_dataset, batch_size=8, shuffle=False)
    logging.info("Filtered uncertain data for GAT evaluation.")

# Step 3: Evaluate GAT model on uncertain cases
gat_predictions, gat_labels = evaluate_gat(gat_loader_filtered, gat_model, device)
logging.info("GAT model evaluation on uncertain data completed.")

# Step 4: Combine ESM and GAT results
final_predictions = combine_results_with_validation(
    esm_results_dict, gat_predictions, gat_labels, uncertain_df, esm_results_dict["uncertain_indices"], gat_threshold
)


# Log uncertain sequence analysis
#logging.info(f"Uncertain sequences: {len(esm_results['uncertain_indices'])}")
logging.info(f"Final predictions updated using GAT validation.")
# Step 5: Calculate final metrics
final_labels = np.array(esm_results_dict["labels"])
final_predictions = np.array(final_predictions)

# Binarize predictions
binary_predictions = np.zeros_like(final_predictions)
binary_predictions[final_predictions >= esm_threshold_high] = 1
binary_predictions[final_predictions <= esm_threshold_low] = 0

# Calculate label-wise MCC
mcc_per_label = [
    matthews_corrcoef(final_labels[:, i], binary_predictions[:, i])
    for i in range(len(label_columns))
]

# Calculate Precision, Recall, F1-Score
precision, recall, f1_scores, _ = precision_recall_fscore_support(
    final_labels, binary_predictions, average=None, zero_division=1
)

# Macro and Micro Averages
macro_f1 = precision_recall_fscore_support(
    final_labels, binary_predictions, average="macro", zero_division=1
)[2]
micro_f1 = precision_recall_fscore_support(
    final_labels, binary_predictions, average="micro", zero_division=1
)[2]

# Accuracy
accuracy = accuracy_score(final_labels, binary_predictions)

# Jaccard Index
jaccard_index = jaccard_score(final_labels, binary_predictions, average="samples")

# Overall MCC
overall_mcc = matthews_corrcoef(final_labels.flatten(), binary_predictions.flatten())

# Average number of predicted labels per sample
avg_predicted_labels = np.sum(binary_predictions, axis=1).mean()

# ROC AUC (Micro)
roc_auc = roc_auc_score(final_labels, final_predictions, average="micro", multi_class="ovo")

# Create metrics dictionary
metrics = {
    "Macro F1": macro_f1,
    "Micro F1": micro_f1,
    "Accuracy": accuracy,
    "Jaccard Index": jaccard_index,
    "Overall MCC": overall_mcc,
    "Avg Predicted Labels": avg_predicted_labels,
    "ROC AUC (Micro)": roc_auc,
    "Precision Per Label": dict(zip(label_columns, precision)),
    "Recall Per Label": dict(zip(label_columns, recall)),
    "F1 Per Label": dict(zip(label_columns, f1_scores)),
    "MCC Per Label": dict(zip(label_columns, mcc_per_label)),
}

# Log final metrics
logging.info(f"Final Metrics: {json.dumps(metrics, indent=4)}")
