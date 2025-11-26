import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional

# --- Configuration ---
ROOT_DIR = '/home/sreekanthvs/my_reserach_works/AUTOMATIC_AGING/RESULTS/TCN_1D_features' 
OUTPUT_PLOT_DIR = '/home/sreekanthvs/my_reserach_works/AUTOMATIC_AGING/RESULTS/TSNE_plots' 
# The script will now look for files with this extension
FEATURE_FILE_EXTENSION = '.npy' 

# Define file names for feature caching (for the aggregated matrix)
FEATURE_CACHE_DIR = '/home/sreekanthvs/my_reserach_works/AUTOMATIC_AGING/RESULTS/TCN_1D_features'
FEATURE_FILE = os.path.join(FEATURE_CACHE_DIR, 'scaled_features_agg.npy')
FULL_LABEL_FILE = os.path.join(FEATURE_CACHE_DIR, 'full_labels_agg.npy')
PLOT_LABEL_FILE = os.path.join(FEATURE_CACHE_DIR, 'plot_labels_agg.npy')

# ------------------------------------------------------------------------------------------------------------------------------------------------

def load_features_from_directory(root_dir: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Traverses the directory, loads individual .npy feature files, and creates labels.
    """
    all_features = []
    all_labels = [] 
    
    print(f"Traversing {root_dir} and loading {FEATURE_FILE_EXTENSION} feature files...")
    
    for dirpath, _, filenames in os.walk(root_dir):
        # The folder name is the primary label
        folder_label = os.path.basename(dirpath)
        if not folder_label: 
            continue
            
        for filename in filenames:
            if filename.endswith(FEATURE_FILE_EXTENSION):
                filepath = os.path.join(dirpath, filename)
                
                try:
                    # Load the feature vector (expected to be a 1D array)
                    features = np.load(filepath)
                    
                    if features.ndim > 1 and features.shape[0] != 1:
                        # Ensures we only load feature vectors, not entire matrices
                        print(f"Skipping {filename}: Not a 1D feature vector.")
                        continue
                        
                    # Flatten the feature vector if it's (1, N) or (N,)
                    features = features.flatten()
                        
                    all_features.append(features)
                    # Label format: "Folder_Name / Filename"
                    file_label = f"{folder_label} / {filename}"
                    all_labels.append(file_label)
                    
                except Exception as e:
                    print(f"Error loading {filepath}: {e}")
                    
    if not all_features:
        print(f"No {FEATURE_FILE_EXTENSION} files found or loaded successfully.")
        return None, None, None

    # Check for consistent feature size
    feature_size = all_features[0].shape[0]
    if not all(f.shape[0] == feature_size for f in all_features):
        print("🛑 Error: Loaded feature vectors have inconsistent lengths. Cannot proceed.")
        return None, None, None
        
    features_matrix = np.array(all_features, dtype=np.float32)
    full_labels = np.array(all_labels, dtype=object) 
    # Simplified label for coloring the plot
    plot_labels = np.array([label.split(' / ')[0] for label in full_labels], dtype=object)
    
    return features_matrix, full_labels, plot_labels

def plot_tsne_results(X_2d: np.ndarray, plot_labels: np.ndarray, output_dir: str):
    """
    Plots the 2D t-SNE results.
    """
    os.makedirs(output_dir, exist_ok=True)
    plot_filepath = os.path.join(output_dir, 'ecg_tsne_plot_npy_direct.png')
    
    plt.figure(figsize=(16, 12))
    
    unique_classes = np.unique(plot_labels)
    cmap = plt.cm.get_cmap('tab20', len(unique_classes))
    label_to_color = {label: cmap(i) for i, label in enumerate(unique_classes)}
    
    for i, label in enumerate(unique_classes):
        indices = plot_labels == label
        plt.scatter(X_2d[indices, 0], X_2d[indices, 1], 
                    c=[label_to_color[label]], label=label, 
                    alpha=1, s=50)

    plt.title('t-SNE Visualization of Pre-Calculated Features', fontsize=16)
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.legend(title='Age Group', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.95)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) 

    plt.savefig(plot_filepath, dpi=1200)
    print(f"\n✅ Plot saved to: {plot_filepath}")
    
# --- Main Logic ---

def main():
    
    os.makedirs(FEATURE_CACHE_DIR, exist_ok=True)
    
    # 1. Check for cached aggregated features
    if os.path.exists(FEATURE_FILE) and os.path.exists(FULL_LABEL_FILE) and os.path.exists(PLOT_LABEL_FILE):
        print("Found cached *aggregated* features. Loading data from .npy files...")
        scaled_features = np.load(FEATURE_FILE, allow_pickle=True)
        full_labels = np.load(FULL_LABEL_FILE, allow_pickle=True)
        plot_labels = np.load(PLOT_LABEL_FILE, allow_pickle=True)
        
    else:
        # 2. Load individual feature files (if cache is missing)
        print(f"No aggregated cache found. Loading individual feature files from: {ROOT_DIR}")
        
        features_matrix, full_labels, plot_labels = load_features_from_directory(ROOT_DIR)
        
        if features_matrix is None:
            return

        print(f"Found {features_matrix.shape[0]} feature vectors with {features_matrix.shape[1]} dimensions each.")
        
        # Pre-processing: Standardize the features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features_matrix)
        print("Features scaled.")
        
        # 3. Save aggregated features to .npy files (caching)
        print(f"Saving aggregated features to cache directory: {FEATURE_CACHE_DIR}")
        np.save(FEATURE_FILE, scaled_features)
        np.save(FULL_LABEL_FILE, full_labels)
        np.save(PLOT_LABEL_FILE, plot_labels)
        print("Aggregated features successfully cached.")


    # 4. Apply t-SNE
    print("\nApplying t-SNE for dimensionality reduction...")
    # Perplexity should be less than the number of samples
    perplexity_val = min(30, len(scaled_features) - 1)
    if perplexity_val < 1:
        print("Cannot run t-SNE: Need at least 2 samples.")
        return
        
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_val, n_iter=1000)
    X_2d = tsne.fit_transform(scaled_features)
    print("t-SNE transformation complete.")
    
    # 5. Plot and Save
    plot_tsne_results(X_2d, plot_labels, OUTPUT_PLOT_DIR)

if __name__ == '__main__':
    main()