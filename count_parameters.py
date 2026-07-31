from adult_datasets import make_adult_dataloaders
from model_architectures.vqc import VQC
from quantum_encodings import ENCODINGS


bundle = make_adult_dataloaders(
    train_csv_path="adult_train.csv",
    test_csv_path="adult_test.csv",
    batch_size=32,
)

for encoding_name, encoding_fn in ENCODINGS.items():
    model = VQC(
        input_dim=bundle.input_dim,
        n_qubits=6,
        n_layers=3,
        output_dim=1,
        encoding_fn=encoding_fn,
        encoding_name=encoding_name,
    )

    num_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(f"{encoding_name}: {num_params} trainable parameters")