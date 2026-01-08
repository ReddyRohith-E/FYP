# System Architecture: Static CNN for ASD Detection

This document outlines the architecture of the system defined in `train_static_cnn.py`. The system processes time-series data to classify Autism Spectrum Disorder (ASD).

## Data Processing Pipeline

1.  **Input**: Time series data from CSV files (Shape: Time × Nodes).
2.  **Feature Extraction**:
    - Compute Pearson Correlation Coefficient (PCC) matrix between nodes.
    - Extract the upper triangle of the correlation matrix to remove redundancy and diagonal elements.
    - Flatten to a 1D feature vector.
3.  **Normalization**: Apply `StandardScaler` to normalize features across the dataset.

## CNN Model Architecture

The model is a 1D Convolutional Neural Network (CNN) implemented in TensorFlow/Keras.

```mermaid
graph TD
    subgraph Preprocessing ["Data Preprocessing"]
        raw[("Raw Time Series\n(Time x Nodes)")] -->|Pearson Correlation| corr[("Correlation Matrix\n(Nodes x Nodes)")]
        corr -->|Upper Triangle| flat["Feature Vector\n(N*(N-1)/2 features)"]
        flat -->|StandardScaler| norm["Normalized Features"]
    end

    subgraph Model ["1D CNN Architecture"]
        norm --> reshape["Reshape\n(Features, 1)"]

        %% Block 1
        reshape --> conv1["Conv1D\n16 Filters, Kernel=32, Stride=2\nActivation: ReLU"]
        conv1 --> bn1["BatchNormalization"]
        bn1 --> pool1["MaxPooling1D\nPool Size=2"]
        pool1 --> drop1["Dropout\nRate=0.3"]

        %% Block 2
        drop1 --> conv2["Conv1D\n32 Filters, Kernel=16, Stride=1\nActivation: ReLU"]
        conv2 --> bn2["BatchNormalization"]
        bn2 --> pool2["MaxPooling1D\nPool Size=2"]
        pool2 --> drop2["Dropout\nRate=0.4"]

        %% Dense Layers
        drop2 --> flatten["Flatten"]
        flatten --> dense1["Dense\n64 Units, ReLU\nL2 Regularization (0.01)"]
        dense1 --> drop3["Dropout\nRate=0.5"]
        drop3 --> output["Output Layer\n1 Unit, Sigmoid\n(Binary Classification)"]
    end

    style Preprocessing fill:#f9f,stroke:#333,stroke-width:2px
    style Model fill:#ccf,stroke:#333,stroke-width:2px
```

## Layer Details

| Layer       | Output Shape                  | Parameters | Description                                                   |
| :---------- | :---------------------------- | :--------- | :------------------------------------------------------------ |
| **Input**   | `(Batch, Features)`           | 0          | Flattened upper triangle correlation features.                |
| **Reshape** | `(Batch, Features, 1)`        | 0          | Adds channel dimension for Conv1D.                            |
| **Conv1D**  | `(Batch, NewFeatures, 16)`    | _Variable_ | Features extraction. Kernel size 32, Stride 2.                |
| **Block 1** | `(Batch, PooledFeatures, 16)` | -          | Batch Norm -> MaxPool(2) -> Dropout(0.3).                     |
| **Conv1D**  | `(Batch, NewFeatures, 32)`    | _Variable_ | Deeper features. Kernel size 16, Stride 1.                    |
| **Block 2** | `(Batch, PooledFeatures, 32)` | -          | Batch Norm -> MaxPool(2) -> Dropout(0.4).                     |
| **Dense**   | `(Batch, 64)`                 | _Variable_ | Fully connected layer with L2 regularization.                 |
| **Output**  | `(Batch, 1)`                  | _Variable_ | Sigmoid activation for binary classification (ASD vs Normal). |
