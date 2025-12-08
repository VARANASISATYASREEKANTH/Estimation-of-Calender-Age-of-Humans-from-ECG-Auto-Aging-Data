# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 13:35:48 2025

@author: SREEKANTHVS
"""

import os
import numpy as np
import pandas as pd
import joblib # To load the scikit-learn model and artifacts
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm # For visualizing prediction progress
import os

# --- Configuration (Must match your training script paths) ---
DATA_ROOT = 'C:/my_projects/AUTO_AGING/RESULTS_SVM'
TEST_DIR = "C:/my_projects/AUTO_AGING/temp_TEST_DATA"
OUTPUT_DIR = 'C:/my_projects/AUTO_AGING/RESULTS_SVM'
all_items = os.listdir(TEST_DIR)
for items in all_items:
# Input Artifact Paths (Ensure these files exist in OUTPUT_DIR)
    MODEL_NAME = 'trained_model_SVM_P0.joblib'
    SCALER_NAME = 'feature_scaler_SVM.joblib'
    ENCODER_NAME = 'label_encoder_SVM.joblib'
    
    MODEL_PATH = os.path.join(OUTPUT_DIR, MODEL_NAME)
    SCALER_PATH = os.path.join(OUTPUT_DIR, SCALER_NAME)
    ENCODER_PATH = os.path.join(OUTPUT_DIR, ENCODER_NAME)
    
    # Output CSV Paths
    INFERENCE_CSV = os.path.join(OUTPUT_DIR, 'inference_results_SVM_ALL'+str(items)+'.csv')
    REPORT_CSV = os.path.join(OUTPUT_DIR, 'classification_report_SVM_ALL'+str(items)+'.csv') # NEW REPORT CSV
    
    FEATURE_DIM = 80000 # Must be the exact dimension used during training
    
    # --- 1. Data Loading Helper Function ---
    
    def load_features_from_folder(data_path):
        """
        Loads features from class-named subfolders dynamically.
        Returns 2D features (N_samples, FEATURE_DIM) and true labels.
        """
        all_features = []
        all_labels = []
        all_file_ids = []
        
        print(f"\nTest data loading started from: {data_path}")
        
        class_dirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
        
        # Use tqdm to show overall class loading progress
        for class_name in tqdm(sorted(class_dirs), desc="Loading Test Classes"):
            class_path = os.path.join(data_path, class_name)
            file_list = [f for f in os.listdir(class_path) if f.endswith('.npy')]
            
            for file_name in file_list:
                file_path = os.path.join(class_path, file_name)
                try:
                    features = np.load(file_path)
                    if features.shape[0] == FEATURE_DIM:
                        all_features.append(features)
                        all_labels.append(class_name) 
                        relative_path = os.path.relpath(file_path, DATA_ROOT)
                        all_file_ids.append(relative_path)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    
        if not all_features:
            print("Error: No valid features found in the directory.")
            return np.array([]), np.array([]), []
            
        return np.array(all_features), np.array(all_labels), all_file_ids
    
    # ----------------------------------------------------------------------
    
    def save_classification_report(y_true, y_pred, filename, target_names):
        """
        Generates the classification report, converts it to a DataFrame, 
        and saves it to a CSV file.
        """
        # Generate the report dictionary
        report_dict = classification_report(y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0)
        
        # Convert to DataFrame
        df_report = pd.DataFrame(report_dict).transpose()
        
        # Save to CSV
        df_report.to_csv(filename, index=True)
        
        print(f"\nClassification report saved to '{filename}'")
        print("-" * 30)
        
        # Print the console version for immediate review
        print("--- Classification Report (Test Set Performance) ---")
        print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))
        print("-" * 30)
    
    # --- 2. Main Inference Function ---
    
    def run_inference():
        """
        Loads trained SVM artifacts and performs prediction on test data.
        """
        print("--- Starting SVM Inference Pipeline ---")
        
        # Check if all required files exist
        required_files = [MODEL_PATH, SCALER_PATH, ENCODER_PATH]
        for path in required_files:
            if not os.path.exists(path):
                print(f"Error: Required file not found at {path}. Exiting.")
                print("Please ensure you run the training script first to save all artifacts.")
                return
    
        # 1. Load Pre-trained Artifacts
        try:
            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            label_encoder = joblib.load(ENCODER_PATH)
            print(f"Successfully loaded model, scaler, and encoder.")
        except Exception as e:
            print(f"Fatal Error loading artifacts: {e}. Exiting.")
            return
    
        # 2. Load Test Data
        X_test, y_test_true, test_file_ids = load_features_from_folder(TEST_DIR)
    
        if X_test.shape[0] == 0:
            print("Error: No test data found for inference. Exiting.")
            return
        
        # 3. Apply Scaling
        X_test_scaled = scaler.transform(X_test)
        print(f"\nTest Input Scaled Shape (Samples, Features): {X_test_scaled.shape}")
    
        # 4. Perform Prediction
        print("\nStarting prediction using the trained SVM model...")
        
        y_pred_indices = []
        y_pred_probs = []
        
        # Predict in batches of 1 using tqdm for progress visualization
        for sample in tqdm(X_test_scaled, desc="Predicting Samples", unit="sample"):
            sample = sample.reshape(1, -1) 
            y_pred_indices.append(model.predict(sample)[0])
            y_pred_probs.append(model.predict_proba(sample)[0])
    
        y_pred_indices = np.array(y_pred_indices)
        y_pred_probs = np.array(y_pred_probs)
        
        # Convert integer indices back to original class names
        y_pred_labels = label_encoder.inverse_transform(y_pred_indices)
    
        # 5. Save Detailed Sample Results
        
        results_data = {
            'File_ID': test_file_ids,
            'True_Label': y_test_true,
            'Predicted_Label': y_pred_labels,
            'Predicted_Probability': [y_pred_probs[i, idx] for i, idx in enumerate(y_pred_indices)]
        }
    
        for i, class_name in enumerate(label_encoder.classes_):
            results_data[f'Prob_{class_name}'] = y_pred_probs[:, i]
    
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(INFERENCE_CSV, index=False)
        
        print(f"\nDetailed inference results saved to '{INFERENCE_CSV}'")
        print("-" * 30)
    
        # 6. Generate and Save Classification Report (NEW STEP)
        save_classification_report(
            y_test_true, 
            y_pred_labels, 
            REPORT_CSV, 
            target_names=label_encoder.classes_
        )
    
        print("Inference process complete. ✅")


if __name__ == '__main__':
    run_inference()