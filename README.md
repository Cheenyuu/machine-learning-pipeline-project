# AutoML Data Preprocessing and Model Selection Pipeline

## Overview

This project implements all steps needed to automate a machine learning pipeline. The project preprocesses tabular datasets, performs feature selection, trains multiple models, and automatically selects the best performing model using cross-validation.

The goal of the project is to minimize manual preprocessing and model selection for students who are new to machine learning and data mining.

---

## Features

### Automated Data Cleaning

* Removes constant- value columns
* Handles missing value tokens:
  * [ ? ]
  * [ NA ]
  * [ N/A ]
  * [ null ]
  * [ None ]
  * [ nan ]

### Feature Type Detection

Automatically determines whether a feature should be treated as 

* Continuous
* Categorical

using:

* Numeric ratio checks
* Unique value ratios
* Entropy measurements
* Integer spacing analysis

### Continuous Feature Processing

* Converts values to numeric types
* Fills missing values with the feature mean
* Applies Min-Max normalization

### Categorical Feature Processing

* Binary factor encoding for two-category variables
* One-hot encoding for multi-category variables
* Fills missing values using the mode

### Feature Selection

Uses Mutual Information to evaluate relevance.

Random noise feature is added to be a baseline, any feature found performing worse than random noise is automatically removed.

### Model Selection

#### Classification Models

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier

#### Regression Models

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor

### Evaluation

Models are evaluated using 5-fold cross-validation to determine effectiveness of candidate models. The highest-performing model is automatically selected

## Project Structure

'''text
project/
|
|-- raw-data/
|    └-- dataset.csv
|
|-- app.py
|
└-- README.md

'''

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
Models Evaluated:
------------------------------
Logistic Regression
Cross validation score mean: 0.91

Decision Tree
Cross validation score mean: 0.88

Random Forest
Cross validation score mean: 0.94

Selected model: Random Forest
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
