import torch
from torch.utils.data import Dataset, DataLoader
from transformers import EsmTokenizer, EsmForSequenceClassification
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score, jaccard_score, matthews_corrcoef
import torch.nn as nn
import torch.optim as optim
import logging
import numpy as np
import torch.nn.functional as F
import os
import json
os.makedirs("Models", exist_ok=True)


# Set up logging
logging.basicConfig(filename='deeploc_confidence_train_test_threshold.log', level=logging.INFO, format='%(asctime)s %(message)s', filemode='w')

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Dataset class
class ProteinSequenceDatasetCSV(Dataset):
    def __init__(self, data, label_columns, tokenizer, max_length=1024):
        self.data = data
        self.label_columns = label_columns
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sequences = self.data['Sequence'].tolist()
        self.labels = self.data[self.label_columns].values

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = str(self.sequences[idx])  # Ensure the sequence is a string
        labels = self.labels[idx].astype(float)
        tokens = self.tokenizer(sequence, max_length=self.max_length, truncation=True, padding='max_length', return_tensors="pt")
        input_ids = tokens['input_ids'].squeeze(0)  # Fix squeeze usage for batch dimension
        attention_mask = tokens['attention_mask'].squeeze(0)  # Fix squeeze usage for batch dimension
        label_tensor = torch.tensor(labels, dtype=torch.float32)
        return input_ids, attention_mask, label_tensor




# Preprocessing function
def preprocess_data(df, sequence_column, label_columns=None):
    # Ensure the dataset contains all required columns
    if label_columns:
        for col in label_columns:
            if col not in df.columns:
                df[col] = 0
    
    # Retain only the necessary columns: ACC (optional), labels, and sequence
    df = df[['ACC'] + label_columns + [sequence_column]]

    return df



# Initialize tokenizer
tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
# DataLoader setup
batch_size = 8
# Load datasets
# train_csv_file = 'source_data/GPS_subcellular_location_train.csv'
# test_csv_file = 'source_data/GPS_subcellular_location_test.csv'

train_csv_file = 'deeploc_train.csv'
test_csv_file = 'deeploc_test.csv'
validation_csv_file = 'deeploc_validation.csv'


# Load and preprocess train data
train_data = pd.read_csv(train_csv_file)
# Dynamically determine label columns from train_data
label_columns = [ 'Cytoplasm', 'Nucleus', 'Extracellular', 
                 'Cell membrane', 'Mitochondrion', 'Plastid', 
                 'Endoplasmic reticulum', 'Lysosome/Vacuole', 
                 'Golgi apparatus', 'Peroxisome']

print("Label columns:", label_columns)

# Preprocess train and test datasets
train_data = preprocess_data(train_data, sequence_column='Sequence', label_columns=label_columns)
train_data = train_data.dropna(subset=['Sequence'])  # Drop rows with missing sequences

validation_data = pd.read_csv(validation_csv_file)
validation_data = preprocess_data(validation_data, sequence_column='Sequence', label_columns=label_columns)

validation_dataset = ProteinSequenceDatasetCSV(data=validation_data, label_columns=label_columns, tokenizer=tokenizer)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

# Similarly for the test data
# Load and preprocess test data
test_data = pd.read_csv(test_csv_file)  # <-- This line was missing!
# Preprocess and clean test data
test_data = preprocess_data(test_data, sequence_column='Sequence', label_columns=label_columns)
test_data = test_data.dropna(subset=['Sequence'])  # Drop rows with missing sequences


print("Label columns:", label_columns)


# Create dataset objects
train_dataset = ProteinSequenceDatasetCSV(data=train_data, label_columns=label_columns, tokenizer=tokenizer)



train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = ProteinSequenceDatasetCSV(data=test_data, label_columns=label_columns, tokenizer=tokenizer)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the model
embedding_dim = 1280
base_model = EsmForSequenceClassification.from_pretrained(
    "facebook/esm2_t33_650M_UR50D", num_labels=len(label_columns)
).base_model

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

model = ESMWithAttention(
    base_model=base_model, embedding_dim=embedding_dim, num_labels=len(label_columns)
).to(device)

# Freeze base model layers except the last two
for param in model.base_model.parameters():
    param.requires_grad = False
for name, param in model.named_parameters():
    if "encoder.layer.31" in name or "encoder.layer.32" in name or "classifier" in name:
        param.requires_grad = True

# Optimizer and scheduler
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)

# Loss function
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return torch.mean(F_loss) if self.reduction == 'mean' else torch.sum(F_loss)

criterion = FocalLoss(alpha=1, gamma=2)

# Training function
def fine_tune_model(train_loader, test_loader, model, criterion, optimizer, device, num_epochs=10, scheduler=None):
    best_threshold = None  # Track the best threshold
    best_f1 = 0  # Track the best Macro F1 score
    best_model_path = None  # Track the path of the best model

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        for batch, (input_ids, attention_mask, labels) in enumerate(train_loader):
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            logits = model(input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch % 50 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch}/{len(train_loader)}], Training Loss: {total_loss / (batch + 1):.4f}")

        avg_train_loss = total_loss / len(train_loader)
        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}")

        if scheduler:
            scheduler.step(avg_train_loss)

        # Save model after each epoch
        save_path = f"Models/fine_tuned_esm_attention_model_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

        # Evaluation phase after each epoch
        metrics = evaluate_model_with_mcc(test_loader, model, device)
        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Validation Metrics: {metrics}")

        # Log validation metrics
        log_message = f"Epoch [{epoch+1}/{num_epochs}], Validation Metrics:\n"
        for key, value in metrics.items():
            if key == "MCC Per Label":
                log_message += f"{key}: {value}\n"
            elif isinstance(value, dict):
                log_message += f"{key}: {json.dumps(value, indent=2)}\n"  # Convert nested dict to JSON-like string
            elif value is not None:
                log_message += f"{key}: {value:.4f}\n"
            else:
                log_message += f"{key}: None\n"
        logging.info(log_message)

        # Tune threshold for Macro F1
        thresholds = np.arange(0.1, 1.0, 0.05)  # Finer granularity for threshold tuning
        best_epoch_threshold, best_epoch_f1, threshold_metrics = tune_threshold(test_loader, model, device, thresholds=thresholds)

        # Log the best threshold and associated Macro F1 for the epoch
        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Best Threshold: {best_epoch_threshold:.2f}, Best Macro F1: {best_epoch_f1:.4f}")

        # Save threshold tuning metrics for the epoch
        threshold_results_path = f"Models/threshold_tuning_epoch_{epoch+1}.json"
        with open(threshold_results_path, "w") as f:
            json.dump(threshold_metrics, f, indent=4)
        logging.info(f"Threshold tuning metrics saved to {threshold_results_path}")

        # Check if this epoch produced the best Macro F1 score
        if best_epoch_f1 > best_f1:
            best_f1 = best_epoch_f1
            best_threshold = best_epoch_threshold
            best_model_path = f"Models/best_model.pth"
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"Best model saved to {best_model_path} with Macro F1: {best_f1:.4f} and Threshold: {best_threshold:.2f}")

    print(f"Training complete. Best Macro F1: {best_f1:.4f} at Threshold: {best_threshold:.2f}")
    logging.info(f"Training complete. Best Macro F1: {best_f1:.4f} at Threshold: {best_threshold:.2f}")

    return best_threshold, best_model_path  # Return the best threshold and model path




def tune_threshold(loader, model, device, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.1)  # Test thresholds from 0.1 to 0.9
    best_threshold = None
    best_f1 = 0
    metrics_at_thresholds = {}

    logging.info(f"Starting threshold tuning with thresholds: {thresholds}")

    for threshold in thresholds:
        logging.info(f"Evaluating threshold: {threshold:.2f}")
        metrics = evaluate_model_with_mcc(loader, model, device, threshold=threshold)
        macro_f1 = metrics["Macro F1"]
        metrics_at_thresholds[threshold] = metrics

        # Log metrics for the current threshold
        logging.info(f"Metrics at threshold {threshold:.2f}: {metrics}")

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_threshold = threshold
            logging.info(f"New best threshold found: {best_threshold:.2f} with Macro F1: {best_f1:.4f}")

    # Log final results
    logging.info(f"Threshold tuning completed. Best Threshold: {best_threshold:.2f}, Best Macro F1: {best_f1:.4f}")
    return best_threshold, best_f1, metrics_at_thresholds



# Evaluation function
def evaluate_model_with_mcc(loader, model, device, threshold=0.5):
    model.eval()
    all_labels, all_probabilities = [], []

    # Extend confidence_bins to include additional ranges
    confidence_bins = {
        "95-100": [0, 0],
        "90-95": [0, 0],
        "85-90": [0, 0],
        "80-85": [0, 0],
        "75-80": [0, 0],
        "70-75": [0, 0],
        "65-70": [0, 0],
        "60-65": [0, 0],
        "55-60": [0, 0],
        "50-55": [0, 0],
        "45-50": [0, 0],
        "40-45": [0, 0],
        "35-40": [0, 0],
        "30-35": [0, 0],
        "below_30": [0, 0],
    }

    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            logits = model(input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probabilities.append(probs)
            all_labels.append(labels.cpu().numpy())

            # Analyze confidence intervals
            for i, prob in enumerate(probs):
                for j, conf in enumerate(prob):
                    bin_key = None
                    if conf >= 0.95:
                        bin_key = "95-100"
                    elif conf >= 0.90:
                        bin_key = "90-95"
                    elif conf >= 0.85:
                        bin_key = "85-90"
                    elif conf >= 0.80:
                        bin_key = "80-85"
                    elif conf >= 0.75:
                        bin_key = "75-80"
                    elif conf >= 0.70:
                        bin_key = "70-75"
                    elif conf >= 0.65:
                        bin_key = "65-70"
                    elif conf >= 0.60:
                        bin_key = "60-65"
                    elif conf >= 0.55:
                        bin_key = "55-60"
                    elif conf >= 0.50:
                        bin_key = "50-55"
                    elif conf >= 0.45:
                        bin_key = "45-50"
                    elif conf >= 0.40:
                        bin_key = "40-45"
                    elif conf >= 0.35:
                        bin_key = "35-40"
                    elif conf >= 0.30:
                        bin_key = "30-35"
                    else:
                        bin_key = "below_30"

                    confidence_bins[bin_key][0] += 1  # Increment total count
                    if labels[i][j] != (conf >= threshold):
                        confidence_bins[bin_key][1] += 1  # Increment incorrect count

    all_labels = np.vstack(all_labels)
    all_probabilities = np.vstack(all_probabilities)
    predictions = (all_probabilities >= threshold).astype(int)

    precision, recall, macro_f1, _ = precision_recall_fscore_support(all_labels, predictions, average='macro', zero_division=1)
    micro_f1 = precision_recall_fscore_support(all_labels, predictions, average='micro', zero_division=1)[2]
    jaccard = jaccard_score(all_labels, predictions, average='samples')
    accuracy = accuracy_score(all_labels, predictions)
    mcc_overall = matthews_corrcoef(all_labels.flatten(), predictions.flatten())
    mcc_per_label = [matthews_corrcoef(all_labels[:, i], predictions[:, i]) for i in range(len(label_columns))]

    try:
        roc_auc = roc_auc_score(all_labels, all_probabilities, average='macro')
    except ValueError:
        roc_auc = None

    # Calculate percentages of incorrect predictions for each confidence bin
    confidence_analysis = {
        bin_key: {
            "Total Predictions": values[0],
            "Incorrect Predictions": values[1],
            "Error Rate (%)": (values[1] / values[0]) * 100 if values[0] > 0 else 0,
        }
        for bin_key, values in confidence_bins.items()
    }

    # Log confidence analysis
    logging.info(f"Confidence Analysis: {confidence_analysis}")

    return {
        "Macro Precision": precision,
        "Macro Recall": recall,
        "Macro F1": macro_f1,
        "Micro F1": micro_f1,
        "Jaccard Index": jaccard,
        "Accuracy": accuracy,
        "ROC AUC": roc_auc,
        "Overall MCC": mcc_overall,
        "MCC Per Label": mcc_per_label,
        "Confidence Analysis": confidence_analysis,
    }


num_epochs = 10
# Train and evaluate the model with the new function
best_threshold, best_model_path = fine_tune_model(train_loader, validation_loader, model, criterion, optimizer, device, num_epochs=10, scheduler=scheduler)

# Load the best model saved during training
try:
    model.load_state_dict(torch.load(best_model_path))
    print(f"Successfully loaded the best model from {best_model_path}")
except FileNotFoundError:
    print(f"Error: Best model file not found at {best_model_path}")

# Final evaluation on the test set using the best threshold
metrics = evaluate_model_with_mcc(test_loader, model, device, threshold=best_threshold)
logging.info(f"Final Test Evaluation Metrics: {metrics}")

# Save final metrics to JSON
with open("final_test_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# Convert results to a DataFrame for display
results_df = pd.DataFrame([metrics])
results_df.insert(0, "Test File", [test_csv_file])
results_df["MCC Per Label"] = results_df["MCC Per Label"].apply(lambda x: ", ".join([f"{val:.4f}" for val in x]))

# Save results as a CSV
results_df.to_csv("evaluation_results.csv", index=False)
logging.info(f"Evaluation results saved to 'evaluation_results.csv'")

# Print final evaluation results
print(results_df)
