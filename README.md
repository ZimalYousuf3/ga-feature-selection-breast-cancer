# Genetic Algorithm Feature Selection on Random Forest

A Genetic Algorithm (GA) implementation for automated feature selection, applied to a Random Forest classifier on the **Breast Cancer Wisconsin (UCI)** dataset. The project compares a baseline model trained on all features against a GA-optimized model trained only on the most informative feature subset.

## Overview

Feature selection is a key step in building efficient and interpretable machine learning models. Instead of using every available feature, this project uses a Genetic Algorithm to search the feature space and evolve an optimal subset of features that maximizes Random Forest classification accuracy.

The workflow:
1. Train a **baseline** Random Forest on all 30 features.
2. Run a **Genetic Algorithm** where each chromosome represents a binary feature mask (1 = feature selected, 0 = feature excluded).
3. Evolve the population over multiple generations using selection, crossover, and mutation.
4. Compare the GA-optimized model's accuracy and feature count against the baseline.

## Dataset

**Breast Cancer Wisconsin (Diagnostic) Dataset**, loaded via `sklearn.datasets.load_breast_cancer`.

- 569 samples
- 30 numeric features (computed from digitized images of breast mass cell nuclei)
- Binary target: malignant / benign

## Methodology

### Baseline Model
A `RandomForestClassifier` (100 estimators) is trained on the full 30-feature set to establish a reference accuracy score.

### Genetic Algorithm

| Component | Description |
|---|---|
| **Chromosome representation** | Binary vector of length 30 (one gene per feature) |
| **Population size** | 10 |
| **Generations** | 20 |
| **Fitness function** | Random Forest test accuracy using only the selected features |
| **Selection** | Tournament selection (3 random contenders per round) |
| **Crossover** | Single-point crossover |
| **Mutation** | Bit-flip mutation, rate = 0.1 |

Each generation, the algorithm selects parents, produces offspring via crossover, applies mutation, and evaluates fitness (Random Forest accuracy) to track the best-performing feature subset over time.

## Results

The notebook produces:
- **Baseline accuracy** — Random Forest performance using all 30 features.
- **GA evolution plot** — best fitness (accuracy) per generation, showing convergence over 20 generations.
- **Optimized results** — final selected feature subset, number of features retained, and final accuracy achieved by the GA-optimized model.

*(See the notebook output for exact numeric results — accuracy may vary slightly between runs due to the algorithm's inherent randomness.)*

## Project Structure
├── GA_Feature_Selection_RandomForest_BreastCancer.ipynb # Main notebook
└── README.md

## Requirements

- Python 3.8+
- numpy
- scikit-learn
- matplotlib

Install dependencies:
```bash
pip install numpy scikit-learn matplotlib
```

## Usage

1. Clone the repository:
```bash
   git clone https://github.com/ZimalYousuf3/ga-feature-selection-breast-cancer.git
   cd ga-feature-selection-breast-cancer
```
2. Open the notebook:
```bash
   jupyter notebook GA_Feature_Selection_RandomForest_BreastCancer.ipynb
```
3. Run all cells to reproduce the baseline model, GA evolution, and optimized results.

> **Note:** This implementation does not fix a random seed for the GA's stochastic components (population initialization, selection, crossover, mutation). As a result, re-running the notebook may produce slightly different selected features and accuracy values across runs, while remaining consistent in overall behavior and trend.

## Key Takeaways

- Genetic Algorithms offer a flexible, model-agnostic approach to feature selection without requiring gradient information.
- Reducing the feature set can maintain or improve classification accuracy while lowering model complexity.
- Tournament selection combined with single-point crossover and mutation is sufficient to converge toward strong feature subsets within a small number of generations.

## Author

**Zimal Yousuf**
GitHub: [@ZimalYousuf3](https://github.com/ZimalYousuf3)

## License

This project is open source and available under the [MIT License](LICENSE).
