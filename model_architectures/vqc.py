import torch
import torch.nn as nn
import pennylane as qml

from .base import BaseQuantumModel


class VQC(BaseQuantumModel):
    """
    Variational Quantum Classifier (VQC)

    Structure:
        classical input
        -> projection to qubits
        -> encoding (pluggable)
        -> variational quantum circuit
        -> measurement
        -> classical output layer
    """

    def __init__(
        self,
        input_dim: int,
        n_qubits: int = 4,
        n_layers: int = 2,
        output_dim: int = 1,
        encoding_fn=None,
        device_fn=None,
    ):
        super().__init__(
            input_dim=input_dim,
            n_qubits=n_qubits,
            encoding_fn=encoding_fn,
            device_fn=device_fn,
        )

        self.n_layers = n_layers
        self.output_dim = output_dim

        # Define quantum circuit
        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs, weights):
            # Encoding (pluggable)
            self.encoding_fn(inputs, wires=range(self.n_qubits))

            # Variational layers
            qml.StronglyEntanglingLayers(
                weights,
                wires=range(self.n_qubits)
            )

            # Measure all qubits
            return [
                qml.expval(qml.PauliZ(i))
                for i in range(self.n_qubits)
            ]

        self.circuit = circuit

        # Trainable quantum parameters
        self.q_weights = nn.Parameter(
            0.01 * torch.randn(self.n_layers, self.n_qubits, 3)
        )

        # Classical output layer
        self.output_layer = nn.Linear(self.n_qubits, output_dim)

    
    def forward(self, x: torch.Tensor):
        # Step 1: project + normalize
        x = self.encode(x)

        # Step 2: run quantum circuit per sample
        hidden_repr = torch.stack([
            torch.stack(self.circuit(sample, self.q_weights))
            for sample in x
        ]).float()

        # Step 3: classical output
        logits = self.output_layer(hidden_repr)

        return logits, hidden_repr