import os
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib # For saving the scikit-learn model
import time # To measure training time
from tqdm import tqdm # Import tqdm for progress visualization

# --- Configuration (Minimal Changes) ---
DATA_ROOT = 'classification_data_root'
TRAIN_DIR = "C:/my_projects/AUTO_AGING/train_TCN_1D_features"
TEST_DIR = "C:/my_projects/AUTO_AGING/temp_TEST_DATA"
# Output paths
OUTPUT_DIR = 'C:/my_projects/AUTO_AGING/RESULTS_SVM'
OUTPUT_CSV = os.path.join(OUTPUT_DIR, 'classification_results_SVM_P0.csv')
OUTPUT_MODEL_JOBLIB = os.path.join(OUTPUT_DIR, 'trained_model_SVM_P0.joblib')
FEATURE_DIM = 80000 # Expected size of the feature vector (must be consistent)
RANDOM_SEED = 42

# SVM Hyperparameters (Can be tuned)
SVM_C = 1.0 # Regularization parameter
SVM_KERNEL = 'rbf' # Kernel type: 'linear', 'poly', 'rbf' (Radial Basis Function), 'sigmoid'
SVM_GAMMA = 'scale' # Kernel coefficient for 'rbf', 'poly' and 'sigmoid'

# Set seed for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory set to: {OUTPUT_DIR}")

# ----------------------------------------------------------------------
## 📂 Data Loading with tqdm
# ----------------------------------------------------------------------

def load_features_from_folder(data_path, is_train=True):
    """
    Loads features from class-named subfolders dynamically,
    using tqdm for progress visualization over files.
    """
    all_features = []
    all_labels = []
    all_file_ids = []
    sample_counts = {}

    data_set_name = "Training" if is_train else "Testing"
    print(f"\n{data_set_name} data loading started from: {data_path}")
    
    # Get all class directories
    class_dirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    
    for class_name in sorted(class_dirs):
        class_path = os.path.join(data_path, class_name)
        file_list = [f for f in os.listdir(class_path) if f.endswith('.npy')]
        
        current_class_count = 0
        
        # Use tqdm to show progress for files within each class
        progress_bar = tqdm(file_list, desc=f"Loading Class: {class_name}", unit="file")
        
        for file_name in progress_bar:
            file_path = os.path.join(class_path, file_name)
            
            try:
                features = np.load(file_path)
                
                # Sanity check for feature dimension consistency
                if features.shape[0] != FEATURE_DIM:
                    # Tqdm will print this without breaking the line
                    progress_bar.write(f"Warning: Skipping {file_path}. Expected dim {FEATURE_DIM}, got {features.shape[0]}")
                    continue
                        
                all_features.append(features)
                all_labels.append(class_name) 
                relative_path = os.path.relpath(file_path, DATA_ROOT)
                all_file_ids.append(relative_path)
                current_class_count += 1
            except Exception as e:
                progress_bar.write(f"Error loading {file_path}: {e}")

        if current_class_count > 0:
            sample_counts[class_name] = current_class_count
            print(f"  -> Loaded {current_class_count} features for class: {class_name}")

    if not all_features:
        print("Error: No valid features found in the directory.")
        return np.array([]), np.array([]), [], {}
        
    print(f"Dynamic class counts for '{os.path.basename(data_path)}': {sample_counts}")
    return np.array(all_features), np.array(all_labels), all_file_ids, sample_counts

# ----------------------------------------------------------------------
## 🧠 SVM Model Functions
# ----------------------------------------------------------------------

def train_svm_model(X_train_scaled, y_train_encoded_indices):
    """
    Creates, trains, and returns a scikit-learn Support Vector Classifier (SVC).
    """
    print("\nCreating and training the Support Vector Machine (SVC) model...")
    
    model = SVC(
        C=SVM_C, 
        kernel=SVM_KERNEL, 
        gamma=SVM_GAMMA, 
        random_state=RANDOM_SEED, 
        probability=True,
        verbose=False 
    )

    # --- Training with Time Measurement ---
    print("SVM training started...")
    start_time = time.time()
    
    # SVC training is a single, computationally intensive step
    model.fit(X_train_scaled, y_train_encoded_indices)
    
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"Training complete. Time taken: {training_time:.2f} seconds.")
    
    return model

def save_trained_model(model, filename):
    """Saves the trained scikit-learn model using joblib."""
    print(f"\nSaving trained SVM model to '{filename}'...")
    try:
        joblib.dump(model, filename)
        print("Model saved successfully.")
    except Exception as e:
        print(f"Error saving model: {e}")
    print("-" * 30)

def test_and_save_results(model, X_test_scaled, y_test_true, test_file_ids, label_encoder):
    """
    Performs classification on the test set and saves the results to a CSV file.
    """
    print("\nClassifying test data and saving results...")
    
    # Predict with progress bar over the samples
    X_test_list = np.array_split(X_test_scaled, X_test_scaled.shape[0]) # Split into one sample per item
    
    y_pred_indices = []
    y_pred_probs = []

    # Iterate through samples with tqdm for prediction progress
    for sample in tqdm(X_test_list, desc="Predicting Test Samples", unit="sample"):
        y_pred_indices.append(model.predict(sample)[0])
        y_pred_probs.append(model.predict_proba(sample)[0])

    y_pred_indices = np.array(y_pred_indices)
    y_pred_probs = np.array(y_pred_probs)
    y_pred_labels = label_encoder.inverse_transform(y_pred_indices)

    # Get the predicted probability for the predicted class
    results_data = {
        'File_ID': test_file_ids,
        'True_Label': y_test_true,
        'Predicted_Label': y_pred_labels,
        'Predicted_Probability': [y_pred_probs[i, idx] for i, idx in enumerate(y_pred_indices)]
    }

    # Add probability for each class
    for i, class_name in enumerate(label_encoder.classes_):
        results_data[f'Prob_{class_name}'] = y_pred_probs[:, i]

    results_df = pd.DataFrame(results_data)
    results_df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\nClassification results saved to '{OUTPUT_CSV}'")
    print("-" * 30)

    print("--- Classification Report (Test Set) ---")
    print(classification_report(y_test_true, y_pred_labels, zero_division=0))
    print("-" * 30)

# ----------------------------------------------------------------------
## 🚀 Main Execution
# ----------------------------------------------------------------------

def main():
    """
    Main function to execute the classification pipeline using SVM with tqdm.
    """
    
    # 1. Load Train Data (uses tqdm inside load_features_from_folder)
    X_train, y_train, _, train_counts = load_features_from_folder(TRAIN_DIR, is_train=True)
    
    if X_train.shape[0] == 0:
        print("Error: No training data found. Exiting.")
        return

    # 2. Label Encoding
    label_encoder = LabelEncoder()
    y_train_encoded_indices = label_encoder.fit_transform(y_train)
    num_classes = len(label_encoder.classes_)
    
    # 3. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    print(f"\nTrain Input Shape (Samples, Features): {X_train.shape}")
    print(f"Number of classes: {num_classes}")
    print(f"Found classes: {label_encoder.classes_}")
    print("-" * 30)

    # 4. Create and Train Model (time is measured inside)
    model = train_svm_model(X_train_scaled, y_train_encoded_indices)
    
    # Save the trained model
    save_trained_model(model, OUTPUT_MODEL_JOBLIB)

    # 5. Load Test Data (uses tqdm inside load_features_from_folder)
    X_test, y_test_true, test_file_ids, test_counts = load_features_from_folder(TEST_DIR, is_train=False)

    if X_test.shape[0] == 0:
        print("Error: No test data found. Exiting.")
        return
    
    # 6. Apply the SAME Scaling to Test Data
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nTest Input Shape (Samples, Features): {X_test.shape}")
    print("-" * 30)

    # 7. Test and Save Results (uses tqdm for prediction progress)
    test_and_save_results(model, X_test_scaled, y_test_true, test_file_ids, label_encoder)
    
    print("Process complete.")


if __name__ == '__main__':
    # Add TensorFlow seed setting here if needed, but for pure SVM it's redundant
    # tf.random.set_seed(RANDOM_SEED) 
    main()