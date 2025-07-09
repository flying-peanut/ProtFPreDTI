# ProtFPreDTI: A Protein Feature-Based Predictor for Drug-Target Interaction

## Table of Contents

- [About The Project](#about-the-project)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)

## About The Project

**ProtFPreDTI** is a computational model designed to predict potential interactions between drugs and protein targets. 

This project provides a GUI integrated with models, which can preprocess and predict data in real time.

## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

This project requires Python 3.8 or later. The necessary Python packages can be installed via `pip` using the provided `requirements.txt` file.

* **Python 3.8+**
* **pip**

### Environment Configuration

1. **Clone the repository:**

   ```sh
   git clone [https://github.com/flying-peanut/ProtFPreDTI.git]
   cd ProtFPreDTI
   ```

2. **Install the required packages:**
   The dependencies are listed in the `requirements.txt` file.

   ```sh
   pip install -r requirements.txt
   ```

   **Key Dependencies:**

   * `pytorch`
   * `numpy`
   * `pandas`
   * `scikit-learn`
   * `rdkit-pypi` 
   * `joblib` 
   * `shap`

   *Note: Please refer to the `requirements.txt` file for a complete and version-specific list of dependencies.*

## Usage

Once the environment is set up, you can use the model for predictions.

**1. Prepare your input data:**

   -   Ensure your target sequences.
   -   Ensure your drug sequences are in SMILES format.
   -   Prepare a CSV file containing smiles sequences, target sequences, and labels.

**2. Run a prediction:**

   ```sh
python predict.py
   ```
