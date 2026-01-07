import pandas as pd
import numpy as np
import os
import glob
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, regularizers
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Ensure reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print(f"TensorFlow Version: {tf.__version__}")

# Configuration
WINDOW_SIZE = 64   # Length of each time segment (e.g., 64 time points)
STRIDE = 32        # Overlap (50% overlap)
BATCH_SIZE = 32
EPOCHS = 50

# Use absolute path or relative to script location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, 'Dataset')
asd_path = os.path.join(dataset_path, 'Training Data', 'ASD')
normal_path = os.path.join(dataset_path, 'Training Data', 'Normal')

def create_windows(data, window_size, stride):
    """Generates sliding windows from time-series data."""
    windows = []
    for i in range(0, len(data) - window_size + 1, stride):
        window = data[i:i + window_size]
        windows.append(window)
    return np.array(windows)

def load_and_augment_data(folder_path, label, window_size, stride):
    """
    Loads CSV files, standardizes them, and applies sliding window augmentation.
    IMPORTANT: We keep track of subject IDs to ensure proper splitting later.
    """
    X = []
    y = []
    groups = []  # To track which subject a window belongs to
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder not found: {folder_path}")
        return np.array([]), np.array([]), np.array([])
    
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    
    print(f"Processing {len(csv_files)} files in {folder_path}...")
    
    if len(csv_files) == 0:
        print("Warning: No CSV files found.")
        return np.array([]), np.array([]), np.array([])
    
    for idx, csv_file in enumerate(csv_files):
        try:
            df = pd.read_csv(csv_file)
            # Drop non-feature columns if any (like Unnamed indices)
            if 'Unnamed: 0' in df.columns:
                df = df.drop(columns=['Unnamed: 0'])
            
            # Simple check for numeric data
            if not np.issubdtype(df.values.dtype, np.number):
                 print(f"Warning: Non-numeric data in {os.path.basename(csv_file)}")
                 continue

            # IMPORTANT: Standardization per subject is often good for fMRI
            # to handle inter-subject variability in signal amplitude
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(df.values)
            
            # Create windows
            subject_windows = create_windows(data_scaled, window_size, stride)
            
            if len(subject_windows) > 0:
                X.append(subject_windows)
                y.extend([label] * len(subject_windows))
                groups.extend([idx] * len(subject_windows)) # Track subject ID
            else:
                 print(f"Warning: File {os.path.basename(csv_file)} resulted in 0 windows (too short?).")
                
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

def build_hybrid_model(input_shape):
    model = models.Sequential()
    
    # CNN Block
    model.add(layers.Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Dropout(0.3))
    
    # LSTM Block
    # return_sequences=False because we want the final state for classification
    model.add(layers.LSTM(64, return_sequences=False))
    model.add(layers.Dropout(0.4))
    
    # Dense Classification Block
    model.add(layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    optimizer = optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def main():
    # Load Data
    print("Loading ASD Data...")
    X_asd, y_asd, g_asd = load_and_augment_data(asd_path, 1, WINDOW_SIZE, STRIDE)

    print("\nLoading Normal Data...")
    X_norm, y_norm, g_norm = load_and_augment_data(normal_path, 0, WINDOW_SIZE, STRIDE)

    # Handle case where one or both are empty for safety
    if len(X_asd) > 0 and len(X_norm) > 0:
        # Adjust group IDs for Normal data so they don't overlap with ASD IDs
        g_norm = g_norm + (g_asd.max() + 1)

        # Combine
        X = np.concatenate([X_asd, X_norm], axis=0)
        y = np.concatenate([y_asd, y_norm], axis=0)
        groups = np.concatenate([g_asd, g_norm], axis=0)

        print("\nData Shapes:")
        print(f"X (features): {X.shape}")
        print(f"y (labels):   {y.shape}")
    else:
        print("\nError: Could not load both classes needed for training.")
        return

    # Improved Splitting Strategy: GroupKFold or Manual Group Split
    # First Split: Train+Val vs Test
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_val_idx, test_idx = next(gss.split(X, y, groups))

    X_train_val, X_test = X[train_val_idx], X[test_idx]
    y_train_val, y_test = y[train_val_idx], y[test_idx]
    groups_train_val = groups[train_val_idx]

    # Second Split: Train vs Val (from the Train+Val set)
    # Using 0.25 of Train+Val to get approx 0.2 of total (0.8 * 0.25 = 0.2)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, val_idx = next(gss_val.split(X_train_val, y_train_val, groups_train_val))

    X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
    y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]

    print(f"Training Set:   {X_train.shape}")
    print(f"Validation Set: {X_val.shape}")
    print(f"Testing Set:    {X_test.shape}")

    # Class Balance Check
    print(f"Train Balance (ASD/Total): {np.sum(y_train)/len(y_train):.2f}")
    print(f"Val Balance (ASD/Total):   {np.sum(y_val)/len(y_val):.2f}")
    print(f"Test Balance (ASD/Total):  {np.sum(y_test)/len(y_test):.2f}")

    # Dynamic Input Shape based upon loaded data
    # X.shape is (Samples, TimeSteps, Features)
    input_shape = (WINDOW_SIZE, X.shape[2]) 
    print(f"Model Input Shape: {input_shape}")

    model = build_hybrid_model(input_shape)
    model.summary()

    # Callbacks - REMOVED EarlyStopping
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001
    )

    # Train
    print("\nStarting Training for 50 Epochs (No Early Stopping)...")
    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[reduce_lr],
        verbose=1
    )

    # Save Model
    save_path = os.path.join(BASE_DIR, 'hybrid_asd_model.h5')
    model.save(save_path)
    print(f"\nModel saved to {save_path}")

    # Plot training history
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plot_path = os.path.join(BASE_DIR, 'training_history.png')
    plt.savefig(plot_path)
    print(f"Training history plot saved to {plot_path}")

    # Evaluation on Test set
    print("\nEvaluating on Test Set...")
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Metrics
    print("Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'ASD']))

    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    try:
        print(f"AUC-ROC Score: {roc_auc_score(y_test, y_pred_prob):.4f}")
    except Exception as e:
        print(f"Could not calculate AUC-ROC: {e}")

if __name__ == "__main__":
    main()
