# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 10:49:48 2025

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
from glob import glob
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
)

# --- GLOBAL CONFIGURATION ---
FEATURE_DIM = 80000           # Must match the dimension of your features
NUM_CLASSES =15             # Must match the number of class folders (3 in dummy data)
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
EPOCHS = 20

# Directory Paths
TRAIN_DATA_DIR = 'C:/my_projects/AUTO_AGING/train_TCN_1D_features'
VAL_DATA_DIR = 'C:/my_projects/AUTO_AGING/val_features'
TEST_DATA_DIR = 'C:/my_projects/AUTO_AGING/temp_TEST_DATA_1'

# Output Paths
MODEL_SAVE_PATH = 'C:/my_projects/AUTO_AGING/RESULTS_MLP/classification_model.pth'
PLOT_SAVE_PATH = 'C:/my_projects/AUTO_AGING/RESULTS_MLP/training_plots.png'
RESULTS_CSV_PATH = 'C:/my_projects/AUTO_AGING/RESULTS_MLP/test_results_detail.csv'
METRICS_CSV_PATH = 'C:/my_projects/AUTO_AGING/RESULTS_MLP/test_metrics_summary.csv'

# --- 1. Custom Dataset Class ---
class FeatureDataset(Dataset):
    """
    Loads features from class-named subfolders.
    """
    def __init__(self, data_dir):
        
        # 1. Map class names (folder names) to numerical labels
        class_names = sorted([d for d in os.listdir(data_dir) 
                              if os.path.isdir(os.path.join(data_dir, d))])
        
        if not class_names:
            raise FileNotFoundError(f"No class folders found in {data_dir}. Check directory structure.")
            
        self.class_to_label = {name: i for i, name in enumerate(class_names)}
        self.label_to_class = {i: name for name, i in self.class_to_label.items()}
        
        print(f"[{os.path.basename(data_dir)}] Detected classes: {self.class_to_label}")
        
        # 2. Collect all file paths and their corresponding labels
        self.data_paths = []
        self.labels = []
        self.class_names_list = [] # Store original class name strings
        
        for class_name, label in self.class_to_label.items():
            class_folder = os.path.join(data_dir, class_name)
            feature_files = glob(os.path.join(class_folder, '*.npy'))
            
            for file_path in feature_files:
                self.data_paths.append(file_path)
                self.labels.append(label)
                self.class_names_list.append(class_name)

        if not self.data_paths:
            raise FileNotFoundError(f"No .npy features found in subfolders of {data_dir}.")

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        # Load the feature from the file path
        feature = np.load(self.data_paths[idx])
        
        # Ensure the feature is a single vector (D,)
        if feature.ndim > 1 and feature.shape[0] == 1:
            feature = feature.squeeze(0)
            
        # Get the corresponding label
        label = self.labels[idx]

        # Convert to PyTorch tensors
        feature_tensor = torch.tensor(feature, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        # Return feature, label, and the original file path/class name for reporting
        return feature_tensor, label_tensor, self.data_paths[idx]

# --- 2. Simple Classification Network (MLP) ---
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
        x = self.layer3(x) # Output logits
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
def train_model():
    print("\n--- Starting Model Training ---")
    
    # Setup data loaders
    train_dataset = FeatureDataset(TRAIN_DATA_DIR)
    val_dataset = FeatureDataset(VAL_DATA_DIR)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Setup model, loss, and optimizer
    model = ClassificationNet(FEATURE_DIM, NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        # Training phase
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

        # Record history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_accuracy)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_accuracy)

        print(f"Epoch {epoch+1:02d}/{EPOCHS}: "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

    # Save the model and plots
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nModel saved to {MODEL_SAVE_PATH}")
    plot_metrics(history)
    print(f"Plots saved to {PLOT_SAVE_PATH}")

    # Return the validation dataset for accessing class mapping
    return val_dataset 

# --- 5. Inference Function ---
def run_inference(dataset):
    print("\n--- Starting Model Inference and Evaluation ---")
    
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"Error: Trained model file not found at {MODEL_SAVE_PATH}. Run training first.")
        return

    # Setup model and load weights
    model = ClassificationNet(FEATURE_DIM, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval() 

    # Setup data loader
    test_dataset = FeatureDataset(TEST_DATA_DIR)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_paths = []
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for features, labels, paths in test_loader:
            outputs = model(features)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)

            all_paths.extend(paths)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())
            
    # Map numerical labels back to class names for readability
    label_to_class_map = dataset.label_to_class 
    true_class_names = [label_to_class_map[l] for l in all_labels]
    pred_class_names = [label_to_class_map[p] for p in all_preds]

    # --- Save Detailed Results to CSV ---
    
    results_df = pd.DataFrame({
        'FilePath': all_paths,
        'True_Label_ID': all_labels,
        'True_Class': true_class_names,
        'Predicted_Label_ID': all_preds,
        'Predicted_Class': pred_class_names
    })
    
    # Add probability columns dynamically
    prob_cols = [f'Prob_{label_to_class_map[i]}' for i in range(NUM_CLASSES)]
    probs_df = pd.DataFrame(np.array(all_probs), columns=prob_cols)
    results_df = pd.concat([results_df, probs_df], axis=1)
    
    results_df.to_csv(RESULTS_CSV_PATH, index=False)
    print(f"\nDetailed inference results saved to {RESULTS_CSV_PATH}")
    
    # --- Calculate and Save Test Metrics Summary ---
    
    overall_accuracy = accuracy_score(all_labels, all_preds)
    
    # Classification Report
    report_dict = classification_report(all_labels, all_preds, 
                                        target_names=dataset.class_to_label.keys(), 
                                        output_dict=True, zero_division=0)
    
    # Extract metrics from the report dictionary
    metrics_list = []
    for class_name, metrics in report_dict.items():
        if class_name in label_to_class_map.values(): # Individual classes
            metrics_list.append({
                'Metric': f'{class_name} Precision', 'Value': metrics['precision']
            })
            metrics_list.append({
                'Metric': f'{class_name} Recall', 'Value': metrics['recall']
            })
            metrics_list.append({
                'Metric': f'{class_name} F1-Score', 'Value': metrics['f1-score']
            })

    metrics_list.append({'Metric': 'Overall Accuracy', 'Value': overall_accuracy})
    metrics_list.append({'Metric': 'Macro Avg F1-Score', 'Value': report_dict['macro avg']['f1-score']})

    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(METRICS_CSV_PATH, index=False)
    print(f"Test metrics summary saved to {METRICS_CSV_PATH}")
    
    # Print a summary
    print("\n--- Summary Classification Report ---")
    print(classification_report(all_labels, all_preds, 
                                target_names=dataset.class_to_label.keys(), zero_division=0))
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    # Ensure FEATURE_DIM and NUM_CLASSES are correct and dummy data is created.
    
    # 1. Train the model and get the dataset object for class mapping
    val_dataset = train_model()

    # 2. Run inference using the trained model
    run_inference(val_dataset)