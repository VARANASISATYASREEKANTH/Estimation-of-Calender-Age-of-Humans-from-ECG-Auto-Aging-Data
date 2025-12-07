# -*- coding: utf-8 -*-
"""
Created on Sun Dec  7 12:16:29 2025

@author: SREEKANTHVS
"""

import os
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf # Explicitly import tensorflow
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
DATA_ROOT = 'classification_data_root'
TRAIN_DIR = "C:/my_projects/AUTO_AGING/train_TCN_1D_features"
TEST_DIR = "C:/my_projects/AUTO_AGING/test_data_features_PTB"#os.path.join(DATA_ROOT, 'test_data_features')

# Output paths
OUTPUT_DIR = 'C:/my_projects/AUTO_AGING/RESULTS'
OUTPUT_CSV = os.path.join(OUTPUT_DIR, 'classification_results.csv')
OUTPUT_PLOT_PNG = os.path.join(OUTPUT_DIR, 'training_history.png')
OUTPUT_MODEL_H5 = os.path.join(OUTPUT_DIR, 'C:/my_projects/AUTO_AGING/RESULTS/trained_model.h5')

FEATURE_DIM = 80000 
RANDOM_SEED = 42
EPOCHS = 200
# OPTIMIZATION 1: Increased BATCH_SIZE significantly for better GPU utilization
# A larger batch size keeps the GPU busy and improves throughput.
BATCH_SIZE =8 # Increased from 4
CLASS_NAMES = [] 

# Set seed for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED) # Set TensorFlow seed

# --- GPU Initialization Check ---
def initialize_gpu():
    """Checks for and initializes GPU devices."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Set memory growth to true to avoid allocating all memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ Found and configured {len(gpus)} GPU(s).")
            # Specify the first GPU for consistency, although TensorFlow usually handles it.
            return tf.device('/gpu:0')
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
            return tf.device('/cpu:0')
    else:
        print("⚠️ No GPU detected. Running on CPU.")
        return tf.device('/cpu:0')
# -----------------------------------


def create_dummy_data_structure():
    """
    Simulates the local folder structure and creates dummy feature files
    (NumPy arrays) for demonstration purposes. 
    """
    print("Setting up dummy data structure with randomized sample sizes...")
    
    random_counts = {
        name: random.randint(50, 150) for name in CLASS_NAMES
    }
    
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for class_name, num_samples in random_counts.items():
        train_class_path = os.path.join(TRAIN_DIR, class_name)
        test_class_path = os.path.join(TEST_DIR, class_name)
        os.makedirs(train_class_path, exist_ok=True)
        os.makedirs(test_class_path, exist_ok=True)

        for i in range(num_samples):
            features_train = np.random.rand(FEATURE_DIM).astype(np.float32)
            np.save(os.path.join(train_class_path, f'{class_name}_{i:03d}.npy'), features_train)

            if i % 2 == 0: 
                features_test = np.random.normal(loc=1.0, scale=0.5, size=FEATURE_DIM).astype(np.float32)
                np.save(os.path.join(test_class_path, f'test_{class_name}_{i:03d}.npy'), features_test)
            
        print(f"Generated {num_samples} total samples for class '{class_name}'.")

    print(f"Dummy data created in directories like '{TRAIN_DIR}'")
    print("-" * 30)


def load_features_from_folder(data_path):
    """
    Loads features from class-named subfolders dynamically.
    """
    all_features = []
    all_labels = []
    all_file_ids = []
    sample_counts = {}

    for class_name in os.listdir(data_path):
        class_path = os.path.join(data_path, class_name)
        
        if not os.path.isdir(class_path):
            continue

        current_class_count = 0
        
        for file_name in os.listdir(class_path):
            if file_name.endswith('.npy'):
                file_path = os.path.join(class_path, file_name)
                
                try:
                    features = np.load(file_path)
                    if features.shape[0] != FEATURE_DIM:
                        print(f"Warning: Skipping {file_name}. Expected dim {FEATURE_DIM}, got {features.shape[0]}.")
                        continue
                    
                    all_features.append(features)
                    all_labels.append(class_name) 
                    relative_path = os.path.join(os.path.basename(data_path), class_name, file_name)
                    all_file_ids.append(relative_path)
                    current_class_count += 1
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        
        if current_class_count > 0:
            sample_counts[class_name] = current_class_count
            print(f"Loaded {current_class_count} features for class: {class_name}")

    print(f"Dynamic class counts for '{os.path.basename(data_path)}': {sample_counts}")
    return np.array(all_features), np.array(all_labels), all_file_ids, sample_counts


def create_and_train_model(X_train, y_train_encoded, input_dim, num_classes):
    """
    Creates, trains, and returns a Keras model and its training history.
    """
    print("Creating and training the Classification Network...")
    
    # --- Model Definition ---
    # OPTIMIZATION 2: Using tf.float32 for consistency (default for NumPy and GPU)
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,), dtype=tf.float32), 
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # --- Training ---
    history = model.fit(X_train, y_train_encoded, 
                        epochs=EPOCHS, 
                        batch_size=BATCH_SIZE, # Uses the optimized BATCH_SIZE
                        validation_split=0.2, 
                        verbose=2) 
    
    print("Training complete.")
    model.summary()
    print("-" * 30)
    return model, history


def save_trained_model(model, filename):
    """
    Saves the trained Keras model to the specified H5 file path.
    """
    print(f"Saving trained model to '{filename}'...")
    try:
        # Saving the entire model (architecture, weights, optimizer state)
        model.save(filename) 
        print("Model saved successfully.")
    except Exception as e:
        print(f"Error saving model: {e}")
    print("-" * 30)


def plot_training_history(history, filename):
    """
    Plots the training and validation loss and accuracy and saves to a PNG file.
    """
    print(f"Generating training history plots and saving to '{filename}'...")
    
    hist = history.history
    epochs = range(1, len(hist['loss']) + 1)

    plt.figure(figsize=(12, 4))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, hist['loss'], 'r', label='Training Loss')
    plt.plot(epochs, hist['val_loss'], 'b', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, hist['accuracy'], 'r', label='Training Accuracy')
    plt.plot(epochs, hist['val_accuracy'], 'b', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    print(f"Plots saved successfully.")
    print("-" * 30)


def test_and_save_results(model, X_test, y_test_true, test_file_ids, label_encoder):
    """
    Performs classification on the test set and saves the results to a CSV file.
    """
    print("Classifying test data and saving results...")
    
    # 1. Get predictions (probabilities)
    y_pred_probs = model.predict(X_test, verbose=0)
    
    # 2. Get predicted labels (index of max probability)
    y_pred_indices = np.argmax(y_pred_probs, axis=1)
    
    # 3. Decode the predicted indices back to class names
    y_pred_labels = label_encoder.inverse_transform(y_pred_indices)

    # 4. Prepare data for the CSV
    results_data = {
        'File_ID': test_file_ids,
        'True_Label': y_test_true,
        'Predicted_Label': y_pred_labels,
        'Predicted_Probability': [y_pred_probs[i, idx] for i, idx in enumerate(y_pred_indices)]
    }

    # Add probabilities for all classes
    for i, class_name in enumerate(label_encoder.classes_):
        results_data[f'Prob_{class_name}'] = y_pred_probs[:, i]

    results_df = pd.DataFrame(results_data)
    
    # 5. Save to CSV
    results_df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"Classification results saved to '{OUTPUT_CSV}'")
    print("-" * 30)

    # Optional: Print evaluation metrics
    print("--- Classification Report (Test Set) ---")
    print(classification_report(y_test_true, y_pred_labels, zero_division=0))
    print("-" * 30)


def main():
    """
    Main function to execute the classification pipeline within a GPU context.
    """
    # OPTIMIZATION 3: Place model definition and training inside the GPU device context
    device_context = initialize_gpu()

    with device_context:
        # 1. Simulate data environment
        create_dummy_data_structure()

        # 2. Load Train Data
        X_train, y_train, _, train_counts = load_features_from_folder(TRAIN_DIR)
        
        if X_train.shape[0] == 0:
            print("Error: No training data found. Exiting.")
            return

        # 3. Label Encoding and One-Hot Encoding
        label_encoder = LabelEncoder()
        y_train_encoded_indices = label_encoder.fit_transform(y_train)
        y_train_encoded = to_categorical(y_train_encoded_indices)
        
        input_dim = X_train.shape[1]
        num_classes = len(label_encoder.classes_)
        print(f"Features dimension: {input_dim}")
        print(f"Total training samples loaded: {X_train.shape[0]}")
        print(f"Number of classes: {num_classes}")
        print("-" * 30)

        # 4. Create and Train Model
        model, history = create_and_train_model(X_train, y_train_encoded, input_dim, num_classes)

        # 5. Save the trained model
        save_trained_model(model, OUTPUT_MODEL_H5)

    # 6. Plot Training History (Can be outside the device context)
    plot_training_history(history, OUTPUT_PLOT_PNG) 

    # 7. Load Test Data
    X_test, y_test_true, test_file_ids, test_counts = load_features_from_folder(TEST_DIR)

    if X_test.shape[0] == 0:
        print("Error: No test data found. Proceeding with saved model.")
        return

    print(f"Total test samples loaded: {X_test.shape[0]}")
    print("-" * 30)
    
    # 8. Test and Save Results (This should also ideally run on the GPU)
    with device_context:
        test_and_save_results(model, X_test, y_test_true, test_file_ids, label_encoder)
    
    print("Process complete.")


if __name__ == '__main__':
    main()