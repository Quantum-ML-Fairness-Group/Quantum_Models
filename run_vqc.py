from datasets import make_compas_dataloaders
from model_architectures.vqc import VQC
from train import train_model
from evaluate import evaluate_model
from utils import save_results_to_csv

# Load dataset
bundle = make_compas_dataloaders(
    "compas-scores-two-years.csv",
    batch_size=32,
)

# Create model
model = VQC(
    input_dim=bundle.input_dim,
    n_qubits=6,
    n_layers=3,
    output_dim=1,
)

# Train
train_model(
    model=model,
    train_loader=bundle.train_loader,
    val_loader=bundle.test_loader,
    epochs=20,
    lr=1e-3,
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
    model_name="VQC",
    accuracy=results["accuracy"],
    dpd=results["demographic_parity_difference"],
)