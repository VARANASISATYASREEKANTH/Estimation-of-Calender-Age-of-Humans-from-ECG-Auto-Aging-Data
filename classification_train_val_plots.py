import os
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report

# --- Configuration ---
# NOTE: Using a single directory for data files to simplify relative paths 
# when referencing data in the CSV output.
DATA_ROOT = 'classification_data_root'
TRAIN_DIR = "C:/my_projects/AUTO_AGING/train_TCN_1D_features"
TEST_DIR = "C:/my_projects/AUTO_AGING/test_data_features_PTB"#os.path.join(DATA_ROOT, 'test_data_features')

# Output paths
OUTPUT_DIR = 'C:/my_projects/AUTO_AGING/RESULTS'
OUTPUT_CSV = os.path.join(OUTPUT_DIR, 'classification_results.csv')
OUTPUT_PLOT_PNG = os.path.join(OUTPUT_DIR, 'training_history.png')
OUTPUT_MODEL_H5 = os.path.join(OUTPUT_DIR, 'C:/my_projects/AUTO_AGING/RESULTS/trained_model.h5')
FEATURE_DIM = 80000 # Size of the feature vector (e.g., from an image or audio embedding)
RANDOM_SEED = 42
EPOCHS = 200 # Increased epochs for better plot illustration
BATCH_SIZE =4

# Default classes for dummy data generation
CLASS_NAMES = [] 

# Set seed for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
# TensorFlow seeds are often complex and depend on system/GPU, but setting general ones helps.
# tf.random.set_seed(RANDOM_SEED) # Assuming TensorFlow is imported as 'tf' implicitly here

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
def create_dummy_data_structure():
    """
    Simulates the local folder structure and creates dummy feature files
    (NumPy arrays) for demonstration purposes. 
    """
    print("Setting up dummy data structure with randomized sample sizes...")
    
    # Generate a random number of samples for each class (e.g., between 50 and 150)
    random_counts = {
        name: random.randint(50, 150) for name in CLASS_NAMES
    }
    
    # Ensure root and output directories exist
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


    # Iterate over the defined class names and their respective random sample counts
    for class_name, num_samples in random_counts.items():
        train_class_path = os.path.join(TRAIN_DIR, class_name)
        test_class_path = os.path.join(TEST_DIR, class_name)
        os.makedirs(train_class_path, exist_ok=True)
        os.makedirs(test_class_path, exist_ok=True)

        # Generate features
        for i in range(num_samples):
            # Training data
            features_train = np.random.rand(FEATURE_DIM).astype(np.float32)
            np.save(os.path.join(train_class_path, f'{class_name}_{i:03d}.npy'), features_train)

            # Testing data 
            # Note: We use a subset of samples for testing for simplicity
            if i % 2 == 0: 
                features_test = np.random.normal(loc=1.0, scale=0.5, size=FEATURE_DIM).astype(np.float32)
                np.save(os.path.join(test_class_path, f'test_{class_name}_{i:03d}.npy'), features_test)
            
        print(f"Generated {num_samples} total samples for class '{class_name}'.")


    print(f"Dummy data created in '{DATA_ROOT}/'")
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
                    all_features.append(features)
                    all_labels.append(class_name) 
                    # Use the file path relative to the DATA_ROOT for unique ID
                    relative_path = os.path.relpath(file_path, DATA_ROOT)
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
    The training set is internally split into train/validation.
    """
    print("Creating and training the Classification Network...")
    
    # --- Model Definition ---
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)), 
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # --- Training ---
    # Using 20% of the training data for validation
    history = model.fit(X_train, y_train_encoded, 
                        epochs=EPOCHS, 
                        batch_size=BATCH_SIZE, 
                        validation_split=0.2, # Added validation split
                        verbose=2) # Set verbose to 2 for line-per-epoch updates
    
    print("Training complete.")
    model.summary()
    print("-" * 30)
    return model, history


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
        # Add probability for the predicted class
        'Predicted_Probability': [y_pred_probs[i, idx] for i, idx in enumerate(y_pred_indices)]
    }

    # Add probabilities for all classes
    # Check if a class exists in the true labels before trying to access its probability
    for i, class_name in enumerate(label_encoder.classes_):
        results_data[f'Prob_{class_name}'] = y_pred_probs[:, i]

    results_df = pd.DataFrame(results_data)
    
    # 5. Save to CSV
    results_df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"Classification results saved to '{OUTPUT_CSV}'")
    print("-" * 30)

    # Optional: Print evaluation metrics (since test labels were available in this example)
    print("--- Classification Report (Test Set) ---")
    print(classification_report(y_test_true, y_pred_labels, zero_division=0))
    print("-" * 30)


def main():
    """
    Main function to execute the classification pipeline.
    """
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
    print(f"Training sample distribution: {train_counts}") 
    print(f"Number of classes: {num_classes}")
    print(f"Found classes: {label_encoder.classes_}")
    print("-" * 30)

    # 4. Create and Train Model
    # Now returns model and history object
    model, history = create_and_train_model(X_train, y_train_encoded, input_dim, num_classes)

    # 5. Plot Training History
    plot_training_history(history, OUTPUT_PLOT_PNG) # Saved plots

    # 6. Load Test Data
    X_test, y_test_true, test_file_ids, test_counts = load_features_from_folder(TEST_DIR)

    if X_test.shape[0] == 0:
        print("Error: No test data found. Exiting.")
        return

    print(f"Total test samples loaded: {X_test.shape[0]}")
    print(f"Test sample distribution: {test_counts}")
    print("-" * 30)


    # 7. Test and Save Results
    test_and_save_results(model, X_test, y_test_true, test_file_ids, label_encoder)
    
    print("Process complete.")


if __name__ == '__main__':
    # Ensure the script runs only when executed directly
    main()