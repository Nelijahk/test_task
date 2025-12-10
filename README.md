## Task 2. Regression on the tabular data

The development process consists of two main stages:

1.  **Exploratory Data Analysis (`task2_notebook.ipynb`):**
    The project started with an exploratory data analysis and initial model experiments in the Jupyter Notebook. This is where the data structure was analysed, and the model architecture was validated.

2.  **Production Pipeline (`train.py` & `predict.py`):**
    Based on the notebook results, the logic was refactored into two executable Python scripts for a clean and modular workflow.
    * All file paths and model hyperparameters are centralised in `config.py`.
    * The scripts rely on `train.csv` and `hidden_test.csv`, which are already included in the root directory. If you wish to use another data, update the paths in `config.py` accordingly.

### Pre-generated Artifacts (Ready to Run)

To make it easier to test the project without retraining from scratch, I have already included the generated artifacts in the repository:

* **`reg_model_weights.pth`**: The trained model weights.
* **`scaler.pkl`**: The fitted MinMaxScaler.
* **`predictions.csv`**: The final output file.

You can run `predict.py` immediately, and it will use these existing files.

---

## Setup and Installation

1.  **Clone the repository:**
    ```
    git clone https://github.com/Nelijahk/test_task.git
    cd test_task\task2
    ```

2.  **Install dependencies:**
    The project uses PyTorch, Pandas, Scikit-learn, and Joblib. Install them using:
    ```
    pip install -r requirements.txt
    ```
---

## Usage

### 1. Training (`train.py`)

Run this script if you want to retrain the model on the data.
It reads the data from `train.csv`, trains the model using parameters from `config.py`, and generates/overwrites two files:
* `reg_model_weights.pth`
* `scaler.pkl`

```
python train.py
```

### 2. Prediction (`predict.py`)

Run this script to generate predictions for the test dataset (`hidden_test.csv`). It automatically loads the saved model weights and the scaler, processes the test data, and saves the results.

```
python predict.py
```

**Output:** The script will create/overwrite `predictions.csv` with the predicted target values.

---
