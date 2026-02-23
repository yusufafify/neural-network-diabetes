"""
model.py — Class-Based Neural Network with Dropout for Diabetes Classification
"""
from typing import List

import tensorflow as tf
from tensorflow.keras import layers, models


class DiabetesNN(tf.keras.Model):
    """
    A feed-forward neural network with Dropout to prevent overfitting.

    Parameters
    ----------
    input_dim : int
        Number of input features (default: 8).
    hidden_units : list[int]
        Neurons per hidden layer (default: [32, 16]).
    dropout_rate : float
        Fraction of units to drop (0.0 to 1.0, default: 0.2).
    activation : str
        Activation function for hidden layers (default: 'relu').
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_units: List[int] | None = None,
        dropout_rate: float = 0.2,
        activation: str = "relu",
        **kwargs,
    ):
        super().__init__(**kwargs)

        if hidden_units is None:
            hidden_units = [32, 16]

        # ── Build the Sequential block internally ─────────────────────
        self.model_layers = models.Sequential([
            # Input + First Hidden Layer
            layers.Dense(hidden_units[0], activation=activation, input_shape=(input_dim,), name="hidden_1"),
            layers.Dropout(dropout_rate, name="dropout_1"),
            
            # Second Hidden Layer
            layers.Dense(hidden_units[1], activation=activation, name="hidden_2"),
            layers.Dropout(dropout_rate, name="dropout_2"),
            
            # Output Layer
            layers.Dense(1, activation='sigmoid', name="output")
        ])

    def call(self, inputs, training=None):
        """
        Executes the forward pass. The 'training' flag is automatically
        handled by Keras to ensure Dropout is ONLY active during training.
        """
        return self.model_layers(inputs, training=training)