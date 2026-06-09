# FedMed: Privacy-Preserving Federated Learning for Medical Tabular Data

[![IEEE Paper](https://img.shields.io/badge/Paper-IEEE%20Format-blue)]()
[![Python](https://img.shields.io/badge/Python-3.11-green)]()
[![License](https://img.shields.io/badge/License-MIT-yellow)]()

## Overview
FedMed is a federated learning framework for medical tabular data that introduces **sensitivity-aware adaptive differential privacy**. Each client computes a local sensitivity score, and the server assigns proportional noise multipliers — ensuring high-sensitivity clients get stronger privacy without degrading low-sensitivity clients' utility.

## Novel Contribution
- First paper to adapt DP **noise multipliers** (not just clipping) based on per-client data sensitivity scores in a federated healthcare setting

## Results (UCI Heart Disease, 10 Federated Clients)
| Method | Accuracy | F1 | AUC-ROC | epsilon |
|--------|----------|-----|---------|---------|
| FedAvg (no DP) | 28.33% | 0.3944 | 0.2723 | inf |
| DP-FedAvg (sigma=1.0) | 36.67% | 0.4242 | 0.4096 | 10.19 |
| DP-FedAvg (sigma=1.5) | 31.67% | 0.4058 | 0.3426 | 10.19 |
| **FedMed (ours)** | **36.67%** | **0.4242** | **0.3839** | **10.19** |

## Run
`ash
pip install numpy scikit-learn torch opacus matplotlib pandas
python fedmed2.py
`

## Paper
Full IEEE-format paper included: fedmed_FINAL.tex (compile on Overleaf)

## Author
Atharv Dorle — SRM Institute of Science and Technology
