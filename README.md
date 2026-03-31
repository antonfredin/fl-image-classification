You are finally at the finish line! A strong, professional `README.md` is the perfect way to wrap up a coding project, especially for a university thesis. It tells the examiner (and any future researchers) exactly what your code does and how to run it.

Here is a clean, GitHub-standard `README.md` based directly on your final code and thesis structure. 

```markdown
# Distributed Image Classification with Federated Learning

This repository contains the source code for a BSc thesis evaluating the performance and communication efficiency of Federated Learning (FL). Specifically, it implements the FederatedAveraging (FedAvg) algorithm and evaluates it against a centralized Deep Learning baseline using the MNIST dataset.

**Authors:** Anton Fredin and Marcus Johansson  
**Institution:** KTH Royal Institute of Technology  
**Course:** EF112X (Spring 2026)  

---

## 📌 Project Overview
Federated Learning allows machine learning models to be trained across multiple decentralized edge devices without exchanging local, privacy-sensitive data. This project investigates the inherent trade-offs of the FedAvg algorithm by focusing on three core experiments:

1. **Client Scaling (Exp 1):** Analyzing convergence speed as the total number of participating clients ($K$) increases, diluting the local data.
2. **Data Heterogeneity (Exp 2):** Comparing the performance of Independent and Identically Distributed (IID) data against highly skewed, 2-class Non-IID data to observe weight divergence.
3. **Client Participation Fraction (Exp 3):** Evaluating the theory of diminishing returns by sampling different fractions ($C$) of a massive client pool ($K=600$) per communication round.

## 🛠️ Requirements & Installation
The experiments are implemented in Python. To run the simulation, you will need the following libraries:

* `torch` (PyTorch)
* `torchvision`
* `numpy`
* `matplotlib`

You can install the dependencies via pip:
```bash
pip install torch torchvision numpy matplotlib
```

## 🚀 Usage
To execute the full suite of experiments, simply run the main script. The script automatically detects if a GPU (`cuda` or `mps`) is available and falls back to `cpu` if not.

```bash
python main.py
```
*(Note: The full simulation runs for 100 epochs/rounds and may take a few hours depending on your hardware.)*

## 📁 Output and Results
To prevent memory issues during long training sessions, the script executes silently and saves all outputs directly to an automatically generated `results/` directory.

After the script finishes, the `results/` folder will contain:
* `baseline_overfitting.png`: A plot confirming the baseline model's generalization.
* `exp1_scaling.png`: Accuracy curves for varying client pool sizes.
* `exp2_distribution.png`: Accuracy comparison between IID and Non-IID setups.
* `exp3_participation.png`: Accuracy curves for varying participation fractions ($C$).
* `final_test_accuracy.txt`: The official numerical test accuracies evaluated on an isolated testing dataset.
* `experiment_raw_data.pkl`: A serialized dictionary containing all raw plotting arrays. This allows you to quickly replot or analyze the data later without re-running the 6-hour training simulation.

## 🔬 Methodology Highlights
* **Shared Initialization:** To ensure a scientifically rigorous and fair comparison, all centralized and federated models are initialized from a single, master set of starting weights.
* **Architecture:** A lightweight, custom Convolutional Neural Network (CNN) designed specifically for the MNIST dataset.
* **Deterministic Execution:** The script seeds `numpy` and `torch` to `0` to ensure completely reproducible runs.

## 📜 License & Citation
If you use this code or rely on these findings, please refer to the corresponding BSc thesis report (TRITA-...). 
```