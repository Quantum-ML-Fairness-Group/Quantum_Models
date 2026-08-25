from datasets import make_compas_dataloaders
from model_architectures.vqc import VQC
from train import train_model
from evaluate import evaluate_model
from utils import save_results_to_csv
from quantum_encodings import ENCODINGS
import torch
import os

# Load dataset
bundle = make_compas_dataloaders(
    "compas-scores-two-years.csv",
    batch_size=32,
)

for encoding_name, encoding_fn in ENCODINGS.items():
    print(f"\nRunning VQC with encoding: {encoding_name}")

    # Create model
    model = VQC(
        input_dim=bundle.input_dim,
        n_qubits=6,
        n_layers=3,
        output_dim=1,
        encoding_fn=encoding_fn,
        encoding_name=encoding_name,
    )

    # Train
    train_model(
        model=model,
        train_loader=bundle.train_loader,
        val_loader=bundle.test_loader,
        epochs=50,
        lr=1e-3,
    )

    torch.save(
    model.state_dict(),
    f"saved_models/compas_{encoding_name}.pth"
)

    # Evaluate
    results = evaluate_model(
        model=model,
        data_loader=bundle.test_loader,
    )

    print("\nFinal Results:")
    print(results)

    save_results_to_csv(
        file_path="results.csv",
        model_name=f"VQC_{encoding_name}",
        accuracy=results["accuracy"],
        group_accuracy_0=results["group_accuracy_0"],
        group_accuracy_1=results["group_accuracy_1"],
        dpd=results["demographic_parity_difference"],
        eod=results["equal_opportunity_difference"],
        di=results["disparate_impact"],
    )