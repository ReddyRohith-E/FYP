import pandas as pd
import numpy as np
import os
import glob
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, optimizers
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'Dataset')
ASD_PATH = os.path.join(DATASET_PATH, 'Training Data', 'ASD')
NORMAL_PATH = os.path.join(DATASET_PATH, 'Training Data', 'Normal')

# DFC Configuration
WINDOW_SIZE = 50
STRIDE = 20
EPOCHS_DL = 40
BATCH_SIZE = 16

def load_raw_data(folder_path, label):
    """Loads raw CSVs and returns list of DataFrames and labels."""
    data_list = []
    labels_list = []
    
    if not os.path.exists(folder_path):
        print(f"Error: Folder not found {folder_path}")
        return [], []
    
    # Sort to ensure deterministic order
    csv_files = sorted(glob.glob(os.path.join(folder_path, '*.csv')))
    print(f"Loading {len(csv_files)} files from {os.path.basename(folder_path)}...")
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if 'Unnamed: 0' in df.columns:
                df = df.drop(columns=['Unnamed: 0'])
            
            if np.issubdtype(df.values.dtype, np.number):
                data_list.append(df.values)
                labels_list.append(label)
        except Exception as e:
            print(f"Error loading {os.path.basename(csv_file)}: {e}")
            
    return data_list, labels_list

def compute_static_features(data_list):
    """Computes flattened upper triangle correlation for static models."""
    features = []
    for subject_data in data_list:
        corr_matrix = np.corrcoef(subject_data.T)
        np.nan_to_num(corr_matrix, copy=False)
        rows, cols = np.triu_indices_from(corr_matrix, k=1)
        features.append(corr_matrix[rows, cols])
    return np.array(features)

def compute_dynamic_features(data_list):
    """Computes sliding window correlation matrices for DFC-CNN."""
    X_windows = []
    y_windows = []
    groups = [] # To track which subject a window belongs to
    
    scaler = StandardScaler()
    
    for subj_idx, subject_data in enumerate(data_list):
        # Standardize time series
        data_scaled = scaler.fit_transform(subject_data)
        
        # Sliding Window
        matrices = []
        for i in range(0, len(data_scaled) - WINDOW_SIZE + 1, STRIDE):
            window = data_scaled[i:i + WINDOW_SIZE]
            corr = np.corrcoef(window.T)
            np.nan_to_num(corr, copy=False)
            matrices.append(corr)
            
        if len(matrices) > 0:
            X_windows.extend(matrices)
            # We don't have labels here, we rely on the caller to manage labels/groups
            # But wait, we need to return consistent X and groups
            
    return np.array(X_windows)

# --- Model Builders ---

def build_static_cnn(input_shape):
    model = models.Sequential([
        layers.Reshape((input_shape[0], 1), input_shape=input_shape),
        layers.Conv1D(16, 32, activation='relu', strides=2, padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.3),
        layers.Conv1D(32, 16, activation='relu', strides=1, padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.4),
        layers.Flatten(),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_dfc_cnn(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.4),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    # 1. Load All Data
    print("Loading data...")
    raw_asd, y_asd = load_raw_data(ASD_PATH, 1)
    raw_norm, y_norm = load_raw_data(NORMAL_PATH, 0)
    
    all_raw = raw_asd + raw_norm
    all_y = np.array(y_asd + y_norm)
    
    print(f"Total Subjects: {len(all_raw)}")
    
    # 2. Strict Subject Split
    # Stratified split to ensure balanced classes in test set
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(sss.split(all_raw, all_y))
    
    print(f"Training Subjects: {len(train_idx)}")
    print(f"Testing Subjects: {len(test_idx)}")
    
    # ---------------------------------------------------------
    # PART A: STATIC MODELS (SVM, RF, MLP, Static FC-CNN)
    # ---------------------------------------------------------
    print("\n--- Preparing Static Features ---")
    
    # Get raw data for splits
    raw_train = [all_raw[i] for i in train_idx]
    y_train = all_y[train_idx]
    raw_test = [all_raw[i] for i in test_idx]
    y_test = all_y[test_idx]
    
    # Compute features
    X_train_static = compute_static_features(raw_train)
    X_test_static = compute_static_features(raw_test)
    
    # Scale
    scaler = StandardScaler()
    X_train_static = scaler.fit_transform(X_train_static)
    X_test_static = scaler.transform(X_test_static)
    
    results = {}
    
    # A1. SVM
    print("Training SVM...")
    svm = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
    svm.fit(X_train_static, y_train)
    results['SVM'] = svm.score(X_test_static, y_test)
    
    # A2. Random Forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train_static, y_train)
    results['Random Forest'] = rf.score(X_test_static, y_test)
    
    # A3. MLP
    print("Training MLP...")
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=500, random_state=42)
    mlp.fit(X_train_static, y_train)
    results['MLP'] = mlp.score(X_test_static, y_test)
    
    # A4. Static FC-CNN
    print("Training Static FC-CNN...")
    static_cnn = build_static_cnn(X_train_static.shape[1:])
    # We use validation split from training set for monitoring, keeping test set pure
    static_cnn.fit(X_train_static, y_train, epochs=EPOCHS_DL, batch_size=8, validation_split=0.2, verbose=0)
    loss, acc = static_cnn.evaluate(X_test_static, y_test, verbose=0)
    results['Static FC-CNN'] = acc
    static_cnn.save('final_static_cnn.keras')
    
    # ---------------------------------------------------------
    # PART B: DYNAMIC MODEL (DFC-CNN)
    # ---------------------------------------------------------
    print("\n--- Preparing Dynamic Features (DFC-CNN) ---")
    
    # Helper to windowize a list of subjects
    def windowize_dataset(subject_list, label_list):
        X_w = []
        y_w = []
        groups_w = []
        
        scaler = StandardScaler() # Standardize each subject's time series
        
        for idx, (data, label) in enumerate(zip(subject_list, label_list)):
            data_scaled = scaler.fit_transform(data)
            matrices = []
            for i in range(0, len(data_scaled) - WINDOW_SIZE + 1, STRIDE):
                corr = np.corrcoef(data_scaled[i:i + WINDOW_SIZE].T)
                np.nan_to_num(corr, copy=False)
                matrices.append(corr)
            
            if matrices:
                X_w.extend(matrices)
                y_w.extend([label] * len(matrices))
                groups_w.extend([idx] * len(matrices)) # Subject ID relative to this list
                
        return np.array(X_w), np.array(y_w), np.array(groups_w)

    X_train_dyn, y_train_dyn, _ = windowize_dataset(raw_train, y_train)
    # For testing, we need groups to vote back to subject level
    X_test_dyn, y_test_dyn, groups_test_dyn = windowize_dataset(raw_test, y_test)
    
    # Add channel dim
    X_train_dyn = X_train_dyn[..., np.newaxis]
    X_test_dyn = X_test_dyn[..., np.newaxis]
    
    print("Training DFC-CNN...")
    dfc_cnn = build_dfc_cnn(X_train_dyn.shape[1:])
    dfc_cnn.fit(X_train_dyn, y_train_dyn, epochs=EPOCHS_DL, batch_size=16, validation_split=0.2, verbose=0)
    
    # Subject-level evaluation for DFC
    y_pred_probs = dfc_cnn.predict(X_test_dyn, verbose=0)
    y_pred_wins = (y_pred_probs > 0.5).astype(int).flatten()
    
    subj_preds = []
    subj_true = []
    unique_subjs = np.unique(groups_test_dyn)
    
    for s_id in unique_subjs:
        mask = (groups_test_dyn == s_id)
        # Check majority vote
        mean_pred = np.mean(y_pred_wins[mask])
        final_decision = 1 if mean_pred > 0.5 else 0
        
        subj_preds.append(final_decision)
        # All windows have same label
        subj_true.append(y_test_dyn[mask][0])
        
    results['DFC-CNN'] = accuracy_score(subj_true, subj_preds)
    dfc_cnn.save('final_dfc_cnn.keras')

    # ---------------------------------------------------------
    # PART C: REPORTING
    # ---------------------------------------------------------
    print("\n" + "="*40)
    print("   FINAL MODEL COMPARISON (Unified Split)")
    print("="*40)
    print(f"{'Model':<20} | {'Test Accuracy':<15}")
    print("-" * 38)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    
    output_lines = []
    output_lines.append("FINAL MODEL COMPARISON (Unified Split)")
    output_lines.append("--------------------------------------")
    output_lines.append(f"{'Model':<20} | {'Test Accuracy':<15}")
    
    for model_name, acc in sorted_results:
        print(f"{model_name:<20} | {acc:.4%}")
        output_lines.append(f"{model_name:<20} | {acc:.4%}")
        
    with open('final_comparison_results.txt', 'w') as f:
        f.write('\n'.join(output_lines))
        
if __name__ == "__main__":
    main()
