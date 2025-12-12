# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 11:11:59 2025

@author: SREEKANTHVS
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from glob import glob
from sklearn.metrics import accuracy_score, classification_report
# import argparse # Removed

# --- GLOBAL CONFIGURATION (Direct Paths and Parameters) ---
# --- Paths ---
TRAIN_DATA_DIR = 'C:/my_projects/AUTO_AGING/train_TCN_1D_features'
VAL_DATA_DIR = 'C:/my_projects/AUTO_AGING/val_features'
INFERENCE_DATA_DIR = 'C:/my_projects/AUTO_AGING/test_data_features_PTB' # Directory for bulk inference input

MODEL_SAVE_PATH = 'C:/my_projects/AUTO_AGING/Results_mlp_inf/classification_model.pth'
LABEL_MAP_SAVE_PATH = 'C:/my_projects/AUTO_AGING/Results_mlp_inf/label_map.json'
PLOT_SAVE_PATH = 'C:/my_projects/AUTO_AGING/Results_mlp_inf/training_plots.png'
INFERENCE_CSV_PATH = 'C:/my_projects/AUTO_AGING/Results_mlp_inf/bulk_inference_results.csv'

# --- Hyperparameters ---
FEATURE_DIM = 80000
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
EPOCHS = 20

# --- 1. Custom Dataset Class (For Training/Validation) ---
class FeatureDataset(Dataset):
    """Loads features from class-named subfolders for training/validation."""
    def __init__(self, data_dir):
        
        class_names = sorted([d for d in os.listdir(data_dir) 
                              if os.path.isdir(os.path.join(data_dir, d))])
        
        if not class_names:
            raise FileNotFoundError(f"No class folders found in {data_dir}. Check structure.")
            
        # Map: Class Name (str) -> Numerical Label (int)
        self.class_to_label = {name: i for i, name in enumerate(class_names)}
        # Map: Numerical Label (int) -> Class Name (str)
        self.label_to_class = {i: name for name, i in self.class_to_label.items()}
        
        self.data_paths = []
        self.labels = []
        
        for class_name, label in self.class_to_label.items():
            class_folder = os.path.join(data_dir, class_name)
            feature_files = glob(os.path.join(class_folder, '*.npy'))
            
            for file_path in feature_files:
                self.data_paths.append(file_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        feature = np.load(self.data_paths[idx])
        if feature.ndim > 1:
            feature = feature.squeeze()
            
        label = self.labels[idx]
        
        return (torch.tensor(feature, dtype=torch.float32), 
                torch.tensor(label, dtype=torch.long), 
                self.data_paths[idx])

# --- 2. Classification Network (MLP) ---
class ClassificationNet(nn.Module):
    """A simple Multi-Layer Perceptron for classification."""
    def __init__(self, input_dim, num_classes):
        super(ClassificationNet, self).__init__()
        # 
        self.layer1 = nn.Linear(input_dim, 256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.layer2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.dropout1(self.relu1(self.layer1(x)))
        x = self.relu2(self.layer2(x))
        x = self.layer3(x) 
        return x

# --- 3. Plotting Function ---
def plot_metrics(history):
    """Generates and saves the loss and accuracy plots."""
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Loss Plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Accuracy Plot
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(PLOT_SAVE_PATH)
    plt.close()

# --- 4. Training Function ---
def train_pipeline():
    print("--- Starting Model Training ---")
    
    # 4.1 Load Data and Determine Class Count
    try:
        train_dataset = FeatureDataset(TRAIN_DATA_DIR)
        val_dataset = FeatureDataset(VAL_DATA_DIR)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return

    num_classes = len(val_dataset.label_to_class)
    print(f"Detected {num_classes} classes: {val_dataset.label_to_class}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 4.2 Setup Model, Loss, and Optimizer
    model = ClassificationNet(FEATURE_DIM, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    # 4.3 Training Loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for features, labels, _ in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = accuracy_score(all_labels, all_preds)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for features, labels, _ in val_loader:
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = accuracy_score(val_labels, val_preds)

        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_accuracy)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_accuracy)

        print(f"Epoch {epoch+1:02d}/{EPOCHS}: "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

    # 4.4 Save Artifacts
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nModel saved to {MODEL_SAVE_PATH}")
    
    # Save the label map (label_to_class)
    with open(LABEL_MAP_SAVE_PATH, 'w') as f:
        json.dump(val_dataset.label_to_class, f)
    print(f"Label map saved to {LABEL_MAP_SAVE_PATH}")
    
    plot_metrics(history)
    print(f"Plots saved to {PLOT_SAVE_PATH}")
    
# --- 5. Bulk Inference Function ---
def infer_pipeline():
    print(f"--- Starting Bulk Inference on {INFERENCE_DATA_DIR} ---")
    
    # 5.1 Load Label Map
    if not os.path.exists(LABEL_MAP_SAVE_PATH):
        print(f"Error: Label map file not found at {LABEL_MAP_SAVE_PATH}. Run training first.")
        return
    
    with open(LABEL_MAP_SAVE_PATH, 'r') as f:
        # Load the dictionary and convert keys (labels) back to integers
        label_map_str = json.load(f)
        LABEL_MAP = {int(k): v for k, v in label_map_str.items()}
    
    num_classes = len(LABEL_MAP)
    print(f"Loaded {num_classes} classes: {LABEL_MAP}")

    # 5.2 Load Model
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"Error: Trained model file not found at {MODEL_SAVE_PATH}. Run training first.")
        return

    model = ClassificationNet(FEATURE_DIM, num_classes)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval() 

    # 5.3 Collect Feature Files (Recursive Search)
    feature_files = glob(os.path.join(INFERENCE_DATA_DIR, '**', '*.npy'), recursive=True)
    
    if not feature_files:
        print(f"No .npy files found in {INFERENCE_DATA_DIR} or its subdirectories.")
        return
    
    print(f"Found {len(feature_files)} feature files for inference.")

    # 5.4 Run Prediction and Collect Results
    results = []
    
    for file_path in feature_files:
        try:
            feature = np.load(file_path).squeeze()
            
            if feature.shape[0] != FEATURE_DIM:
                print(f"[Warning] Skipping {file_path}: dim mismatch ({feature.shape[0]} vs {FEATURE_DIM})")
                continue

            feature_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                outputs = model(feature_tensor)
                probabilities = torch.softmax(outputs, dim=1).squeeze(0)
                
                predicted_index = torch.argmax(probabilities).item()
                confidence = probabilities[predicted_index].item()
                predicted_class = LABEL_MAP.get(predicted_index)

            # Prepare row data
            row = {
                'FilePath': os.path.relpath(file_path, INFERENCE_DATA_DIR), 
                'Predicted_ID': predicted_index,
                'Predicted_Class': predicted_class,
                'Confidence': confidence,
            }
            # Add probabilities for all classes
            for i in range(num_classes):
                class_name = LABEL_MAP.get(i)
                row[f'Prob_{class_name}'] = probabilities[i].item()
                
            results.append(row)

        except Exception as e:
            print(f"[Error] Failed to process {file_path}: {e}")

    # 5.5 Save to CSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv(INFERENCE_CSV_PATH, index=False)
        print(f"\nInference complete. Results saved to {INFERENCE_CSV_PATH}")
    else:
        print("\nNo successful predictions to save.")


# --- 6. Main Execution ---
if __name__ == '__main__':
    
    print("Welcome to the Classification Pipeline.")
    print("--- Configuration ---")
    print(f"Training Data Dir: {TRAIN_DATA_DIR}")
    print(f"Inference Data Dir: {INFERENCE_DATA_DIR}")
    print(f"Model Path: {MODEL_SAVE_PATH}")
    print("---------------------")
    
    # You can choose the mode here. 'train' always runs training.
    # 'infer' runs inference on the INFERENCE_DATA_DIR.

    mode = 'infer' # Change this to 'infer' to only run inference

    if mode == 'train':
        train_pipeline()
        # Optional: Run inference automatically after training
        # infer_pipeline() 
    elif mode == 'infer':
        infer_pipeline()
    else:
        print("Invalid mode specified. Set 'mode' to 'train' or 'infer'.")