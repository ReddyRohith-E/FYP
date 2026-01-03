import pandas as pd
import numpy as np
import os
import glob
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Configuration (Must match training)
WINDOW_SIZE = 64
STRIDE = 32
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, 'Dataset')
asd_path = os.path.join(dataset_path, 'Training Data', 'ASD')
normal_path = os.path.join(dataset_path, 'Training Data', 'Normal')

def create_windows(data, window_size, stride):
    windows = []
    for i in range(0, len(data) - window_size + 1, stride):
        window = data[i:i + window_size]
        windows.append(window)
    return np.array(windows)

def load_and_augment_data(folder_path, label, window_size, stride):
    X = []
    y = []
    groups = []
    
    if not os.path.exists(folder_path):
        print(f"Error: Folder not found: {folder_path}")
        return np.array([]), np.array([]), np.array([])
    
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    
    for idx, csv_file in enumerate(csv_files):
        try:
            df = pd.read_csv(csv_file)
            if 'Unnamed: 0' in df.columns:
                df = df.drop(columns=['Unnamed: 0'])
            
            if not np.issubdtype(df.values.dtype, np.number):
                 continue

            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(df.values)
            
            subject_windows = create_windows(data_scaled, window_size, stride)
            
            if len(subject_windows) > 0:
                X.append(subject_windows)
                y.extend([label] * len(subject_windows))
                groups.extend([idx] * len(subject_windows))
                
        except Exception as e:
            print(f"Error loading {os.path.basename(csv_file)}: {e}")
            
    if len(X) > 0:
        X = np.vstack(X)
        y = np.array(y)
        groups = np.array(groups)
    else:
        X = np.array([])
        y = np.array([])
        groups = np.array([])
    
    return X, y, groups

def main():
    # 1. Load Data
    print("Loading Data for Verification...")
    X_asd, y_asd, g_asd = load_and_augment_data(asd_path, 1, WINDOW_SIZE, STRIDE)
    X_norm, y_norm, g_norm = load_and_augment_data(normal_path, 0, WINDOW_SIZE, STRIDE)

    if len(X_asd) == 0 or len(X_norm) == 0:
        print("Error: Failed to load data.")
        return

    # Adjust group IDs
    g_norm = g_norm + (g_asd.max() + 1)

    X = np.concatenate([X_asd, X_norm], axis=0)
    y = np.concatenate([y_asd, y_norm], axis=0)
    groups = np.concatenate([g_asd, g_norm], axis=0)

    # 2. Re-create Split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_val_idx, test_idx = next(gss.split(X, y, groups))
    
    X_test = X[test_idx]
    y_test = y[test_idx]
    
    print(f"Test Set Shape: {X_test.shape}")

    # 3. Load Model
    model_path = os.path.join(BASE_DIR, 'hybrid_asd_model.h5')
    if not os.path.exists(model_path):
        print("Error: Model file not found.")
        return
        
    print(f"Loading model from {model_path}...")
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 4. Evaluate
    print("Evaluating...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # 5. Detailed Report
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'ASD']))

if __name__ == "__main__":
    main()
