from datasets import make_compas_dataloaders
from model_architectures.qnn import QNN
from train import train_model
from evaluate import evaluate_model
from utils import save_results_to_csv
import os
import torch

#Load data
bundle = make_compas_dataloaders(
    "compas-scores-two-years.csv",
    batch_size=32,
)

os.makedirs("saved_models", exist_ok=True)

#Create model
model =QNN(
    input_dim=bundle.input_dim,
    n_qubits=6,
    n_layers=3,
    output_dim=1,
    readout_qubit=0,
)

#Train
train_model(
    model=model,
    train_loader=bundle.train_loader,
    val_loader=bundle.test_loader,
    epochs=50,
    lr=1e-3,
)

torch.save(
    model.state_dict(),
    "saved_models/compas_qnn.pth"
)

# Evaluate
results = evaluate_model(
    model=model,
    data_loader=bundle.test_loader,
)

print("\nFinal Results:")
print(results)

# Save to CSV
save_results_to_csv(
    file_path="results.csv",
    model_name="QNN",
    accuracy=results["accuracy"],
    group_accuracy_0=results["group_accuracy_0"],
    group_accuracy_1=results["group_accuracy_1"],
    dpd=results["demographic_parity_difference"],
    eod=results["equal_opportunity_difference"],
    di=results["disparate_impact"],
)