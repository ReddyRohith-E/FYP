
import argparse
import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def parse_args():
    parser = argparse.ArgumentParser()
    
    # SageMaker passes hyperparameters as CLI args
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    
    # Data directories
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    
    return parser.parse_args()

def load_data(data_dir):
    """
    Loads .npy files from the data directory.
    Returns X (data), y (labels).
    """
    if not data_dir:
        return np.array([]), np.array([])
        
    files = glob.glob(os.path.join(data_dir, "*.npy"))
    data = []
    labels = []
    
    # In a real scenario, labels should be passed via a CSV or manifest.
    # For this POC, we will simulate labels or try to extract from filename if possible.
    # Since we lack a phenotypic file in this simple script, we will generate dummy labels
    # OR you must ensure the phenotypic data is passed to the container.
    # For now: Random labels for demonstration of PIPELINE mechanics (replace with real labels)
    
    print(f"Loading {len(files)} files from {data_dir}...")
    for f in files:
        arr = np.load(f)
        # Add channel dimension: (64, 64, 64) -> (64, 64, 64, 1)
        arr = np.expand_dims(arr, axis=-1)
        data.append(arr)
        # DUMMY LABEL: 0 or 1
        labels.append(np.random.randint(0, 2)) 
        
    return np.array(data), np.array(labels)

def build_3d_cnn(input_shape):
    model = models.Sequential()
    
    # Convolutional Block 1
    model.add(layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(layers.BatchNormalization())
    
    # Convolutional Block 2
    model.add(layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu'))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(layers.BatchNormalization())
    
    # Convolutional Block 3
    model.add(layers.Conv3D(128, kernel_size=(3, 3, 3), activation='relu'))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(layers.BatchNormalization())
    
    # Dense Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    return model

def main():
    args = parse_args()
    
    # Load data
    X_train, y_train = load_data(args.train)
    # X_val, y_val = load_data(args.validation) # Optional if you have validation channel
    
    if len(X_train) == 0:
        print("No training data found. Exiting.")
        return

    print(f"Training Data Shape: {X_train.shape}")
    
    # Build Model
    input_shape = X_train.shape[1:]
    model = build_3d_cnn(input_shape)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
                  
    model.summary()
    
    # Train
    model.fit(X_train, y_train,
              epochs=args.epochs,
              batch_size=args.batch_size,
              validation_split=0.1) # Use split for now if val channel empty
              
    # Save Model
    # Save as TensorFlow SavedModel format in the SM_MODEL_DIR
    model.save(f"{args.model_dir}/1")
    print(f"Model saved to {args.model_dir}/1")

if __name__ == "__main__":
    main()
