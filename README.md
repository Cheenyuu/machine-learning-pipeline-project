# AutoML Feature Selection & Modeling Pipeline

A lightweight AutoML-style pipeline that automatically handles data cleaning, feature type inference, feature selection, encoding, normalization, and model evaluation for both classification and regression problems.

The goal of this project is to take a raw CSV dataset and produce a competitive baseline model with minimal manual preprocessing.

---

## Features

### Automatic Feature Understanding
- Infers feature types (categorical vs continuous)
- Detects unusable or noisy features
- Handles mixed-type columns
- Identifies ID-like or irrelevant columns automatically

---

### Data Cleaning
- Standardizes missing values (`?, NA, null, NaN`, etc.)
- Removes features with more than 50% missing values
- Drops predefined unreliable features (e.g., `id`, `time`, `name`)
- Tracks removed features with explanations

---

### Feature Engineering

#### Continuous Features
- Missing value imputation:
  - Median (if skewed)
  - Mean (otherwise)
- Min-max normalization

#### Categorical Features
- Binary encoding (2 unique values)
- One-hot encoding (less than 100 categories)
- Frequency encoding (high cardinality)

---

### Feature Selection

Uses a combination of statistical and heuristic methods:

- Low variance filtering
- Pearson correlation pruning (continuous features)
- Chi-square independence testing (categorical features)
- Mutual information ranking (supervised selection)
- Random noise baseline thresholding

---

### Model Training and Evaluation

Automatically trains and compares multiple models:

#### Classification
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

#### Regression
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

Evaluation uses 5-fold cross-validation:
- Classification: Accuracy, F1-score
- Regression: R², Mean Squared Error

The best model is selected based on the primary metric (F1 or R²).

---

### High-Dimensional Mode
If dataset complexity is high:
- Automatically triggers fast feature selection
- Re-evaluates model performance after pruning
- Compares performance before and after feature reduction

---

## Project Structure

## Project Structure

```text
project/
|
|-- raw-data/
|    └-- dataset.csv
|
|-- app.py
|
└-- README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/youruser/AutoML-Pipeline.git
cd automl-pipeline
```

Install required packages:
```bash
pip install pandas numpy scipy scikit-learn
```

## Usage

### 1. Add a Dataset

A default dataset is already provided, however, if you would like to add your own, please remove the dataset that is currently there and place a single CSV file inside 'raw_data' directory:

```text
raw_data/
└-- kidney_disease.csv
```

### 2. Specify the Target Variable

Inside 'main()':

```python
target_column = "classification"
```

### Run the Pipeline

```bash
python automl_pipeline.py
```

---

## Pipeline Workflow

```text
Load Dataset
    ↓
Clean Missing Values
    ↓
Detect Feature Types
    ↓
Encode Categorical Features
    ↓
Normalize Continuous Features
    ↓
Feature Selection
    ↓
Train Multiple Models
    ↓
Cross Validation
    ↓
Select Best Model
```

---

## Example Output

```text
Type: regression | Mode; Name: Linear Regression | score: 0.992
```

---

## Current Limitations

* Only supports CSV datasets
* Requires exactly one dataset in the `raw_data` directory
* Does not currently save trained models
* Limited hyperparameter tuning
* Mixed-type columns are dropped rather than repaired

---

## Future Improvements

* Hyperparameter optimization
* Model persistence with Pickle or Joblib
* Feature importance visualization
* Additional machine learning models
* Support for multiple datasets
* Train/test split reporting

---

## Technologies Used

* Python
* Pandas
* NumPy
* SciPy
* Scikit-Learn

---

## Author

**Chenyu Zheng**

B.S. Computer Science
Pennsylvania State University
