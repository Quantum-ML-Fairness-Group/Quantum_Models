import pennylane as qml


def angle_encoding(inputs, wires):
    """
    Basic angle encoding using RY rotations.
    Requires len(inputs) == len(wires).
    """
    qml.AngleEmbedding(
        features=inputs,
        wires=wires,
        rotation="Y",
    )


def entangled_3_layer_encoding(inputs, wires):
    """
    3 layers of angle encoding interleaved with ring entanglement.
    """
    wires = list(wires)

    for _ in range(3):
        qml.AngleEmbedding(
            features=inputs,
            wires=wires,
            rotation="Y",
        )

        for i in range(len(wires) - 1):
            qml.CNOT(wires=[wires[i], wires[i + 1]])

        if len(wires) > 1:
            qml.CNOT(wires=[wires[-1], wires[0]])


def entangled_5_layer_encoding(inputs, wires):
    """
    5 layers of angle encoding interleaved with ring entanglement.
    """
    wires = list(wires)

    for _ in range(5):
        qml.AngleEmbedding(
            features=inputs,
            wires=wires,
            rotation="Y",
        )

        for i in range(len(wires) - 1):
            qml.CNOT(wires=[wires[i], wires[i + 1]])

        if len(wires) > 1:
            qml.CNOT(wires=[wires[-1], wires[0]])


def amplitude_encoding(inputs, wires):
    """
    Amplitude encoding.

    PennyLane will pad with 0s and normalize automatically.
    Requires len(inputs) <= 2 ** len(wires).
    """
    qml.AmplitudeEmbedding(
        features=inputs,
        wires=wires,
        pad_with=0.0,
        normalize=True,
    )


def iqp_encoding(inputs, wires):
    """
    IQP feature-map encoding.
    Requires len(inputs) == len(wires).
    """
    qml.IQPEmbedding(
        features=inputs,
        wires=wires,
        n_repeats=2,
    )


def dense_angle_encoding(inputs, wires):
    """
    Dense angle encoding.

    Uses multiple rotations per feature to pack more information
    into each qubit.
    Requires len(inputs) == len(wires).
    """
    wires = list(wires)

    for i, wire in enumerate(wires):
        qml.RY(inputs[i], wires=wire)
        qml.RZ(inputs[i], wires=wire)
        qml.RX(inputs[i], wires=wire)


ENCODINGS = {
    # "angle": angle_encoding,
    "entangled_3_layer": entangled_3_layer_encoding,
    "entangled_5_layer": entangled_5_layer_encoding,
    # "amplitude": amplitude_encoding,
    "iqp": iqp_encoding,
    "dense_angle": dense_angle_encoding,
}