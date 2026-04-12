import torch
import torch.nn as nn
import pennylane as qml

from .base import BaseQuantumModel


class CCQC(BaseQuantumModel):
    """
    Circuit-Centric Quantum Classifier (CCQC)

    Key idea:
        - shallow (low-depth) circuit
        - readout qubit for prediction
        - NISQ-friendly design
    """

    def __init__(
        self,
        input_dim: int,
        n_qubits: int = 4,
        n_layers: int = 2,
        output_dim: int = 1,
        encoding_fn=None,
        device_fn=None,
        readout_qubit: int = 0,
    ):
        super().__init__(
            input_dim=input_dim,
            n_qubits=n_qubits,
            encoding_fn=encoding_fn,
            device_fn=device_fn,
        )

        self.n_layers = n_layers
        self.readout_qubit = readout_qubit

        # Quantum circuit
        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs, weights):
            # Encoding (pluggable)
            self.encoding_fn(inputs, wires=range(self.n_qubits))

            # Shallow variational circuit
            for l in range(self.n_layers):

                # Single-qubit rotations
                for i in range(self.n_qubits):
                    qml.RY(weights[l, i, 0], wires=i)
                    qml.RZ(weights[l, i, 1], wires=i)

                # Linear entanglement (chain)
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])

            # Measure only readout qubit
            return qml.expval(qml.PauliZ(self.readout_qubit))

        self.circuit = circuit

        # Trainable parameters
        self.q_weights = nn.Parameter(
            0.01 * torch.randn(self.n_layers, self.n_qubits, 2)
        )

        # Classical output layer
        self.output_layer = nn.Linear(1, output_dim)

    def forward(self, x: torch.Tensor):
        # Step 1: projection + normalization
        x = self.encode(x)

        # Step 2: run circuit per sample
        hidden_repr = torch.stack([
            self.circuit(sample, self.q_weights)
            for sample in x
        ]).unsqueeze(1).float()

        # Step 3: classical output
        logits = self.output_layer(hidden_repr)

        return logits, hidden_repr