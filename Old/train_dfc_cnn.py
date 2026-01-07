import pandas as pd
import numpy as np
import os
import glob
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, optimizers
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
WINDOW_SIZE = 50
STRIDE = 20  # Significant overlap to generate more samples
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, 'Dataset')
asd_path = os.path.join(dataset_path, 'Training Data', 'ASD')
normal_path = os.path.join(dataset_path, 'Training Data', 'Normal')
EPOCHS = 25
BATCH_SIZE = 16

def compute_correlation_matrix(window_data):
    """
    Computes Pearson correlation matrix for a time window.
    Input: (Time, Nodes)
    Output: (Nodes, Nodes)
    """
    # Transpose to (Nodes, Time) for corrcoef
    corr_matrix = np.corrcoef(window_data.T)
    # Fill NaN with 0 (in case of constant signal)
    np.nan_to_num(corr_matrix, copy=False)
    return corr_matrix

def create_dynamic_fc(data, window_size, stride):
    """Generates correlation matrices from sliding windows."""
    matrices = []
    # data shape: (Time, Nodes)
    for i in range(0, len(data) - window_size + 1, stride):
        window = data[i:i + window_size]
        fc_matrix = compute_correlation_matrix(window)
        matrices.append(fc_matrix)
    return np.array(matrices)

def load_and_process_data(folder_path, label):
    X = []
    y = []
    groups = []
    
    if not os.path.exists(folder_path):
        print(f"Error: Folder not found {folder_path}")
        return np.array([]), np.array([]), np.array([])
    
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    print(f"Processing {len(csv_files)} files in {os.path.basename(folder_path)}...")
    
    for idx, csv_file in enumerate(csv_files):
        try:
            df = pd.read_csv(csv_file)
            if 'Unnamed: 0' in df.columns:
                df = df.drop(columns=['Unnamed: 0'])
            
            if not np.issubdtype(df.values.dtype, np.number):
                 continue

            # Standardize time series before correlation
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(df.values)
            
            # Generate Dynamic Functional Connectivity Matrices
            subject_matrices = create_dynamic_fc(data_scaled, WINDOW_SIZE, STRIDE)
            
            if len(subject_matrices) > 0:
                X.append(subject_matrices)
                y.extend([label] * len(subject_matrices))
                groups.extend([idx] * len(subject_matrices))
                
        except Exception as e:
            print(f"Error loading {os.path.basename(csv_file)}: {e}")
            
    if len(X) > 0:
        X = np.vstack(X)
        # Add channel dimension for CNN: (Samples, Rows, Cols, Channels)
        X = X[..., np.newaxis]
        y = np.array(y)
        groups = np.array(groups)
    else:
        X = np.array([])
        y = np.array([])
        groups = np.array([])
    
    return X, y, groups

def build_dfc_cnn(input_shape):
    model = models.Sequential()
    
    # 2D Convolutional Layers
    # Brain networks have local structure (nodes are often ordered by atlas regions)
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.GlobalAveragePooling2D()) # Reduce dimensionality effectively
    model.add(layers.Dropout(0.4))
    
    # Dense Classifier
    model.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    optimizer = optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def main():
    # 1. Load Data
    print("Loading ASD Data...")
    X_asd, y_asd, g_asd = load_and_process_data(asd_path, 1)
    
    print("\nLoading Normal Data...")
    X_norm, y_norm, g_norm = load_and_process_data(normal_path, 0)
    
    if len(X_asd) == 0 or len(X_norm) == 0:
        print("Failed to load data.")
        return

    # Adjust group IDs
    g_norm = g_norm + (g_asd.max() + 1)
    
    X = np.concatenate([X_asd, X_norm], axis=0)
    y = np.concatenate([y_asd, y_norm], axis=0)
    groups = np.concatenate([g_asd, g_norm], axis=0)
    
    print(f"\nTotal Data Shape: {X.shape}")
    print(f"Total Labels: {len(y)}")
    
    # 2. Split Data (GroupShuffleSplit to keep subjects intact)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    print(f"Train Shape: {X_train.shape}")
    print(f"Test Shape: {X_test.shape}")
    
    # 3. Model
    input_shape = X_train.shape[1:]
    model = build_dfc_cnn(input_shape)
    model.summary()
    
    # 4. Train
    print("\nStarting Training...")
    # Class weights to handle any imbalance from windowing
    total = len(y_train)
    pos = np.sum(y_train)
    neg = total - pos
    weight_for_0 = (1 / neg) * (total / 2.0)
    weight_for_1 = (1 / pos) * (total / 2.0)
    class_weight = {0: weight_for_0, 1: weight_for_1}

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2, # Simple split on windows for monitoring
        class_weight=class_weight,
        verbose=1
    )
    
    # 5. Evaluate
    print("\nEvaluating on Test Set...")
    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy (Window-level): {acc:.4f}")
    
    # 6. Subject-level Voting (Optional but recommended)
    # We need groups for test set to aggregate
    groups_test = groups[test_idx]
    
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    print("\nWindow-level Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'ASD']))
    
    # Aggregate predictions per subject
    unique_subjects = np.unique(groups_test)
    subj_true = []
    subj_pred = []
    
    for subj_id in unique_subjects:
        # Get all windows for this subject
        mask = (groups_test == subj_id)
        windows_pred = y_pred[mask]
        windows_true = y_test[mask]
        
        # Majority vote
        final_pred = 1 if np.mean(windows_pred) > 0.5 else 0
        final_true = windows_true[0] # All windows have same label
        
        subj_pred.append(final_pred)
        subj_true.append(final_true)
    
    subj_acc = accuracy_score(subj_true, subj_pred)
    subj_report = classification_report(subj_true, subj_pred, target_names=['Normal', 'ASD'])
    
    print(f"\nSubject-level Accuracy: {subj_acc:.4f}")
    print("\nSubject-level Report:")
    print(subj_report)
    
    with open('dfc_results.txt', 'w') as f:
        f.write(f"Subject-level Accuracy: {subj_acc:.4f}\n")
        f.write("Subject-level Report:\n")
        f.write(subj_report)
    
    # Save Model
    model.save('dfc_cnn_model.keras')
    print("Model saved to dfc_cnn_model.keras")

if __name__ == "__main__":
    main()
