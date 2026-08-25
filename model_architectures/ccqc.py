import torch
import torch.nn as nn
import pennylane as qml

from .base import BaseQuantumModel


class CCQC(BaseQuantumModel):
    """
    Circuit-Centric Quantum Classifier (CCQC)

    Design:
        classical input
        -> projection to qubits
        -> feature encoding
        -> compact circuit-centric variational block
        -> single-qubit readout
        -> classical output layer

    The circuit uses parameterized Rot gates together with
    ring entanglement to distinguish it from the QNN architecture.
    """

    def __init__(
        self,
        input_dim: int,
        n_qubits: int = 6,
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
        self.output_dim = output_dim
        self.readout_qubit = readout_qubit

        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs, weights):

            # 1. Encode classical information
            self.encoding_fn(
                inputs,
                wires=range(self.n_qubits)
            )

            # 2. Circuit-centric variational layers
            for layer in range(self.n_layers):

                # General single-qubit rotations
                for qubit in range(self.n_qubits):
                    qml.Rot(
                        weights[layer, qubit, 0],
                        weights[layer, qubit, 1],
                        weights[layer, qubit, 2],
                        wires=qubit,
                    )

                # Ring entanglement
                for qubit in range(self.n_qubits - 1):
                    qml.CNOT(
                        wires=[qubit, qubit + 1]
                    )

                # Close the ring
                if self.n_qubits > 1:
                    qml.CNOT(
                        wires=[self.n_qubits - 1, 0]
                    )

            # 3. Single-qubit readout
            return qml.expval(
                qml.PauliZ(self.readout_qubit)
            )

        self.circuit = circuit

        # Three trainable rotation parameters per qubit per layer
        self.q_weights = nn.Parameter(
            0.01 * torch.randn(
                self.n_layers,
                self.n_qubits,
                3,
            )
        )

        # Convert quantum expectation value into output logit
        self.output_layer = nn.Linear(
            1,
            output_dim
        )

    def forward(self, x: torch.Tensor):

        # Classical projection + normalization
        x = self.encode(x)

        # Quantum circuit for every sample
        hidden_repr = torch.stack([
            self.circuit(sample, self.q_weights)
            for sample in x
        ]).unsqueeze(1).float()

        # Final classification logit
        logits = self.output_layer(hidden_repr)

        return logits, hidden_repr