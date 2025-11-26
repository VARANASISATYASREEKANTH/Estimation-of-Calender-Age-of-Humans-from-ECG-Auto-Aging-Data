import wfdb
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten
import os

# --- Configuration ---
DATA_FOLDER = '/home/sreekanthvs/my_reserach_works/AUTOMATIC_AGING/DATASETS/test_data_ptxbl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/records100/14000' 
OUTPUT_FOLDER = '/home/sreekanthvs/my_reserach_works/AUTOMATIC_AGING/RESULTS/test_data_features/records100/14/'
TARGET_SIGNAL_LENGTH = 5000 # Standard length for CNN input

# --- NEW: Function to generate file list based on .dat files ---
def get_record_names(data_dir):
    """
    Finds all .dat files in the directory and returns their record names 
    (filename without the .dat extension).
    """
    record_names = []
    try:
        # List all files and folders in the directory
        for filename in os.listdir(data_dir):
            # Check if the file ends with .dat
            if filename.endswith('.dat'):
                # Extract the record name by stripping the .dat extension
                record_name = filename[:-4] 
                record_names.append(record_name)
        return record_names
    except FileNotFoundError:
        print(f"Error: The data folder '{data_dir}' was not found.")
        return []

# Generate the list of records to process
FILE_LIST = get_record_names(DATA_FOLDER)
if not FILE_LIST:
    print("No .dat files found. Exiting.")
    # Exit or handle the error appropriately
    # exit() 


# --- 1. Define the 1D Temporal CNN (TCNN) Model ---
def create_feature_extractor(input_length):
    """Defines a simple 1D CNN model for feature extraction."""
    input_shape = (input_length, 1) 
    input_layer = Input(shape=input_shape)
    
    x = Conv1D(filters=32, kernel_size=5, activation='relu', padding='same')(input_layer)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(filters=64, kernel_size=5, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    feature_vector = Flatten(name='feature_output')(x)
    
    model = Model(inputs=input_layer, outputs=feature_vector)
    return model

# --- 2. Signal Preprocessing Function ---
def preprocess_signal(signal, target_length):
    """Standardize the signal and handle length differences (padding/truncating)."""
    if signal.ndim == 0:
        signal = signal.reshape(-1)
        
    # Standard Scaling
    if signal.std() > 0:
        signal = (signal - signal.mean()) / signal.std()
    
    # Padding or Truncating
    current_length = len(signal)
    if current_length > target_length:
        signal = signal[:target_length]
    elif current_length < target_length:
        padding = target_length - current_length
        signal = np.pad(signal, (0, padding), 'constant')
        
    return signal.astype(np.float32)

# --- 3. Main Feature Extraction and Saving Loop ---
def extract_and_save_features_simple(file_list, data_dir, output_dir, target_length):
    
    os.makedirs(output_dir, exist_ok=True)
    
    feature_extractor = create_feature_extractor(target_length)
    print("Feature Extractor Model Summary:")
    feature_extractor.summary()
    
    for filename in file_list: # 'filename' here is actually the record name
        file_path = os.path.join(data_dir, filename)
        
        try:
            # wfdb.rdsamp automatically looks for the .hea and .dat files
            signals, fields = wfdb.rdsamp(file_path)
            channel_names = fields['sig_name']
            
            print(f"\nProcessing file: **{filename}** (Channels: {channel_names})")
            
            file_output_dir = os.path.join(output_dir, filename)
            #os.makedirs(file_output_dir, exist_ok=True)

            # Process each channel individually
            for i, channel_name in enumerate(channel_names):
                single_channel_signal = signals[:, i]
                
                # Preprocess, Reshape, and Extract Features... (same as before)
                preprocessed_signal = preprocess_signal(single_channel_signal, target_length)
                model_input = preprocessed_signal[np.newaxis, :, np.newaxis]
                
                features = feature_extractor.predict(model_input, verbose=0)
                features = features.flatten() 
                
                # Saving features
                feature_filename = f"{filename}_{channel_name}_features.npy"
                save_path = os.path.join(output_dir, feature_filename)
                
                np.save(save_path, features)
                print(f"  - Extracted {len(features)} features for channel **{channel_name}** and saved to: {save_path}")

        except Exception as e:
            print(f"Error processing record {filename}. Is the corresponding .hea file present? Error: {e}")
            continue

# --- Execution ---
if __name__ == "__main__":
    if FILE_LIST:
        print(f"Found {len(FILE_LIST)} records to process based on .dat files.")
        extract_and_save_features_simple(
            file_list=FILE_LIST,
            data_dir=DATA_FOLDER,
            output_dir=OUTPUT_FOLDER,
            target_length=TARGET_SIGNAL_LENGTH
        )
        print("\nFeature extraction process completed.")
    else:
        print("Feature extraction skipped due to no records found.")