Quantum Machine Learning Fairness Experiments

This repository contains experiments for evaluating fairness in different quantum machine learning models and quantum data encoding strategies.

The main experiments compare:
Model architectures: VQC, QNN, and CCQC
Encoding strategies: Angle, Entangled 3-Layer, Entangled 5-Layer, IQP, and Dense Angle
Datasets: COMPAS and Adult
Fairness metrics: Demographic Parity Difference (DPD), Equal Opportunity Difference (EOD), and Disparate Impact (DI)

Requirements
The code is written in Python and uses:
Python 3
PyTorch
PennyLane
NumPy
Pandas
Scikit-learn

Install the required packages with:
pip install torch pennylane numpy pandas scikit-learn

Dataset Files
Place the dataset files in the main project directory.

For COMPAS:
compas-scores-two-years.csv

For Adult:
adult_train.csv
adult_test.csv

Main Files

datasets.py
Loads and preprocesses the COMPAS dataset.

adult_datasets.py
Loads and preprocesses the Adult dataset.

quantum_encodings.py
Contains the quantum encoding strategies used in the experiments:
Angle Encoding
Entangled 3-Layer Encoding
Entangled 5-Layer Encoding
IQP Encoding
Dense Angle Encoding

model_architectures/base.py
Contains the common base class used by the quantum models.

model_architectures/vqc.py
model_architectures/qnn.py
model_architectures/ccqc.py
Contain the different quantum model architectures.

train.py
Contains the model training procedure.

evaluate.py
Evaluates trained models and computes:
Overall accuracy
Group-specific accuracy
Demographic Parity Difference
Equal Opportunity Difference
Disparate Impact

fairness.py
Contains the implementations of the fairness metrics.

utils.py
Contains utility functions for saving experimental results to CSV files.

Running the COMPAS Encoding Experiment
The COMPAS encoding experiment keeps the model fixed as a VQC and compares the different encoding methods.

Run:
python run_vqc.py

The experiment uses:
Dataset: COMPAS
Model: VQC
Qubits: 6
Variational layers: 3
Batch size: 32
Learning rate: 0.001
Epochs: 50
Quantum backend: PennyLane default.qubit

Each encoding is trained separately.

The resulting metrics are saved to:
results.csv

The CSV contains:
model
accuracy
group_accuracy_0
group_accuracy_1
demographic_parity_difference
equal_opportunity_difference
disparate_impact

For COMPAS:
group 0 = Caucasian
group 1 = African-American

Running the Adult Encoding Experiment
Run:
python run_vqc_adult.py

The Adult experiment keeps the VQC architecture fixed and compares the same encoding strategies.

For Adult:
group 0 = Female
group 1 = Male
The results are saved to:
adult_vqc_encoding_results.csv

Running the Architecture Experiments
The architecture experiments compare:
VQC
QNN
CCQC
while keeping the dataset, encoding strategy, and training configuration fixed.

Run each model using its corresponding script, for example:
python run_vqc.py
python run_qnn.py
python run_ccqc.py

The architecture comparison is performed on the COMPAS dataset.

For a fair comparison, the experiments should use the same:
number of qubits
encoding method
train/test split
batch size
optimizer
learning rate
number of epochs
random seed

Saved Models
Trained PyTorch model weights are stored as .pth files.
Example:
saved_models/
    compas_angle.pth
    compas_entangled_3_layer.pth
    compas_entangled_5_layer.pth
    compas_iqp.pth
    compas_dense_angle.pth
    adult_angle.pth
    adult_entangled_3_layer.pth
    adult_entangled_5_layer.pth
    adult_iqp.pth
    adult_dense_angle.pth

Architecture models can also be stored as:
saved_models/
    compas_vqc.pth
    compas_qnn.pth
    compas_ccqc.pth

The .pth files contain the trained model parameters and allow models to be evaluated again without retraining.

Loading a Saved Model
First create the model with the same architecture used during training.
Then load its saved parameters:
model.load_state_dict(
    torch.load("saved_models/compas_vqc.pth")
)

model.eval()
The model can then be evaluated directly:
results = evaluate_model(
    model=model,
    data_loader=bundle.test_loader,
)

Fairness Metrics
Demographic Parity Difference (DPD)
Measures the difference in positive prediction rates between the two demographic groups.
A value closer to:
0
indicates greater demographic parity.

Equal Opportunity Difference (EOD)
Measures the difference in true positive rates between demographic groups.
A value closer to:
0
indicates greater equality of opportunity.

Disparate Impact (DI)
Measures the ratio of positive prediction rates between demographic groups.
A value closer to:
1
indicates greater parity.

Reproducibility
For reproducible experiments, use the same random seeds before model initialization and training:
import random
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

The COMPAS train/test split already uses:
random_state=42

Consistent seeds are important because quantum models can produce different accuracy and fairness results depending on parameter initialization and training order.
Notes

PennyLane may display a warning if NumPy is older than version 2.0. This does not prevent the current experiments from running, but future versions of PennyLane may require NumPy 2.x.

The saved .pth models should generally not be committed directly to GitHub if they are large. They can instead be shared through the team's shared storage folder.