import pandas as pd
import numpy as np
import os
import glob
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, optimizers
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, 'Dataset')
asd_path = os.path.join(dataset_path, 'Training Data', 'ASD')
normal_path = os.path.join(dataset_path, 'Training Data', 'Normal')
EPOCHS = 50
BATCH_SIZE = 8

def compute_upper_triangle(data):
    """
    Computes Pearson correlation and returns flattened upper triangle.
    data shape: (Time, Nodes)
    Output shape: (Features,)
    """
    corr_matrix = np.corrcoef(data.T)
    np.nan_to_num(corr_matrix, copy=False)
    
    # Get upper triangle indices (excluding diagonal)
    rows, cols = np.triu_indices_from(corr_matrix, k=1)
    return corr_matrix[rows, cols]

def load_data(folder_path, label):
    X = []
    y = []
    
    if not os.path.exists(folder_path):
        print(f"Error: Folder not found {folder_path}")
        return np.array([]), np.array([])
    
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    print(f"Processing {len(csv_files)} files in {os.path.basename(folder_path)}...")
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if 'Unnamed: 0' in df.columns:
                df = df.drop(columns=['Unnamed: 0'])
            
            if not np.issubdtype(df.values.dtype, np.number):
                 continue

            # Compute features: Static FC (upper triangle)
            features = compute_upper_triangle(df.values)
            
            X.append(features)
            y.append(label)
                
        except Exception as e:
            print(f"Error loading {os.path.basename(csv_file)}: {e}")
            
    return np.array(X), np.array(y)

def build_static_cnn(input_shape):
    model = models.Sequential()
    
    # 1D CNN Layers
    # Input: (Features, 1)
    model.add(layers.Reshape((input_shape[0], 1), input_shape=input_shape))
    
    model.add(layers.Conv1D(16, 32, activation='relu', strides=2, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Conv1D(32, 16, activation='relu', strides=1, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Dropout(0.4))
    
    model.add(layers.Flatten())
    
    model.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    optimizer = optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def main():
    # 1. Load Data
    print("Loading ASD Data...")
    X_asd, y_asd = load_data(asd_path, 1)
    
    print("\nLoading Normal Data...")
    X_norm, y_norm = load_data(normal_path, 0)
    
    if len(X_asd) == 0 or len(X_norm) == 0:
        print("Failed to load data.")
        return

    X = np.concatenate([X_asd, X_norm], axis=0)
    y = np.concatenate([y_asd, y_norm], axis=0)
    
    # Standardize Features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    print(f"\nTotal Data Shape: {X.shape}")
    
    # 2. Split Data
    # Using StratifiedShuffleSplit for consistent validation
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(sss.split(X, y))
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # 3. Model
    model = build_static_cnn(X_train.shape[1:])
    model.summary()
    
    # 4. Train
    print("\nStarting Training...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        verbose=1
    )
    
    # 5. Evaluate
    print("\nEvaluating on Test Set...")
    loss, acc = model.evaluate(X_test, y_test)
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    print(f"Test Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred, target_names=['Normal', 'ASD'])
    print(report)
    
    # Save results
    with open('static_cnn_results.txt', 'w') as f:
        f.write(f"Test Accuracy: {acc:.4f}\n")
        f.write("Classification Report:\n")
        f.write(report)
        
    model.save('static_cnn_model.keras')
    print("Model saved to static_cnn_model.keras")

if __name__ == "__main__":
    main()
