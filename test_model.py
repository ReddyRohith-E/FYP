
import os
import tensorflow as tf
from train import build_3d_cnn

def test_model_architecture():
    """
    Verifies that the 3D CNN model can be built and has the correct output shape.
    """
    input_shape = (64, 64, 64, 1)
    model = build_3d_cnn(input_shape)
    
    assert model is not None
    assert len(model.layers) > 0
    
    # Check output shape
    # Create a dummy input
    dummy_input = tf.random.normal((1, 64, 64, 64, 1))
    output = model(dummy_input)
    
    # Expect output shape (1, 1) because last layer is Dense(1)
    assert output.shape == (1, 1)
    
    print("Model architecture test passed!")
    model.summary()

if __name__ == "__main__":
    test_model_architecture()
