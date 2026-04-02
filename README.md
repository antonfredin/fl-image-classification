# Distributed Image Classification with Federated Learning

This repository contains the source code for a BSc thesis evaluating the performance and communication efficiency of Federated Learning (FL). Specifically, it implements the FederatedAveraging (FedAvg) algorithm and evaluates it against a centralized Deep Learning baseline using the MNIST dataset.

**Authors:** Anton Fredin and Marcus Johansson  
**Institution:** KTH Royal Institute of Technology  
**Course:** EF112X (Spring 2026)  

---

## Project Overview
Federated Learning allows machine learning models to be trained across multiple decentralized edge devices without exchanging local, privacy-sensitive data. This project investigates the trade-offs of the FedAvg algorithm by focusing on three core experiments:

1. **Client Scaling:** Analyzing convergence speed as the total number of participating clients ($K$) increases.
2. **Data Heterogeneity:** Comparing the performance of IID data against non-IID data.
3. **Client Participation Fraction:** Sampling different fractions ($C$) of a client pool ($K=600$) each communication round.

## Requirements & Usage
The experiments are implemented in Python. To run the simulation, you will need the following libraries:

```bash
pip install torch torchvision numpy matplotlib
```

To execute the experiments, simply run the main script:

```bash
python main.py
```

## License
This codebase was developed as part of a BSc thesis at KTH and is intended for educational and research purposes.
