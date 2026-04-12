import torch
import torch.nn as nn
import pennylane as qml

# Default encoding (to be replaced)
def default_encoding(inputs, wires):
    """
    Default feature map: angle encoding with RY rotations.
    """
    qml.AngleEmbedding(inputs, wires=wires, rotation="Y")

# Default device (to be replaced)
def default_device(n_qubits):
    """
    Default noiseless simulator.
    """
    return qml.device("default.qubit", wires=n_qubits)

# Base class for all architectures
class BaseQuantumModel(nn.Module):
    """
    Base class for all quantum model architectures.

    Provides:
    - input projection to match number of qubits
    - encoding hook (pluggable)
    - device hook (pluggable)
    - common interface for all models
    """

    def __init__(
        self,
        input_dim: int,
        n_qubits: int = 4,
        encoding_fn=None,
        device_fn=None,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.n_qubits = n_qubits

        # Encoding (can be replaced)
        self.encoding_fn = encoding_fn or default_encoding

        # Device (can be replaced for noise experiments)
        self.dev = (device_fn or default_device)(n_qubits)

        # Project classical features → number of qubits
        self.input_proj = nn.Linear(input_dim, n_qubits)

    # Shared preprocessing
    def encode(self, x: torch.Tensor):
        """
        Maps classical input to valid quantum angles.
        """
        x = self.input_proj(x)
        x = torch.tanh(x) * torch.pi

        return x

    # Forward must be implemented
    def forward(self, x):
        raise NotImplementedError(
            "Each model must implement its own forward() method."
        )

def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path, map_location="cpu"):
    model.load_state_dict(torch.load(path, map_location=map_location))
    model.eval()
    return model