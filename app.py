import pandas as pd
import os
import numpy as np

from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from scipy.stats import pearsonr
from scipy.stats import chi2_contingency
from tqdm import tqdm

class feature_metadata:
    def __init__(self, feature_name):
        self.name = feature_name
        self.type = None
        self.var = None
        self.dom = None

    def show_metadata(self):
        print("[]------------------------------------------------[]")
        print(f"feature name: {self.name}\nfeature type: {self.type}")
        print("[]------------------------------------------------[]")
        return

def LOAD_MANUAL_OVERRIDE():
    override_file = open("MANUAL_OVERRIDE_FILE.txt")
    tokens = override_file.read().split()
    return tokens

# GLOBAL VALUES
GARBAGE_TOKENS = ['?', 'NA', 'N/A', 'null', 'None', 'nan', 'NaN', '']
REMOVED_FEATURES = {}
PREDEFINED_REMOVE_FLAGS = ['name', 'time', 'id', 'jersey']
MANUAL_OVERRIDE_TOKENS = LOAD_MANUAL_OVERRIDE()
#feature metadata idea
METADATA = {}

def target_categorical_preprocessing(target_column):
    try:
        target_column, _ = target_column.factorize()
        return target_column
    except Exception as e:
        print(f"Error converting target column to integer value")

def normalization(dataframe, feature):
    series = dataframe[feature]

    min_val = series.min()
    max_val = series.max()

    if min_val == max_val:
        dataframe[feature] = 0.0
        return

    dataframe[feature] = (series - min_val) / (max_val - min_val)

class ManyFilesError(Exception):
    """Too many files within a single folder"""
    pass

def load_data():
    #data collection
    folder_path = "raw_data"
    files = [f for f in os.listdir(folder_path) if f.endswith(".csv") and os.path.isfile(os.path.join(folder_path, f))]
    dataframe = None
    #I might need to write a unique joining algorithm
    for file_name in files:
        #we need to convert them to dataframes as we iterate
        cur_df = None
        try:
            cur_df = pd.read_csv(f"raw_data/{file_name}")
        except Exception as e:
            print(f"Could not read {file_name}")
            continue
        if dataframe is None:
            dataframe = pd.read_csv(f"raw_data/{file_name}")
        else:
            raise ManyFilesError("Only one file in the folder at a time")
    return dataframe

def test_variance(X, feature_name):
    if X[feature_name].var() < 0.01:
        X.drop(columns = [feature_name], inplace = True)
        REMOVED_FEATURES[feature_name] = "below variance threshold"
        #true for removed
        return True
    return False

def test_correlation(X, feature_name):
    #greedy approach- remove any feature that has a high correlation value associated with our current feature
    for feature in tqdm(X.columns, desc = f"correlation test using {feature_name}"):
        #print(f"testing between {feature_name} and {feature}")
        if METADATA[feature].type == "continuous" and feature != feature_name:
            current_feature_data = X[feature_name]
            comparison_feature_data = X[feature]
            corr, p_val = pearsonr(current_feature_data, comparison_feature_data)
            #high correlation
            if corr >= 0.85:
                #print(f"removed {feature}")
                X.drop(columns = [feature], inplace = True)
                REMOVED_FEATURES[feature] = f"high correlation to {feature_name}, corr_value of {corr}"

def test_chi_square(X, feature_name):
    for feature in tqdm(X.columns, desc = f"Chi-square test using {feature_name}"):
        if METADATA[feature].type == "categorical" and feature != feature_name:
            current_feature_data = X[feature_name]
            comparison_feature_data = X[feature]
            contingency_table = pd.crosstab(current_feature_data, comparison_feature_data)
            chi_2, p_val, dof, expected = chi2_contingency(contingency_table)

            if p_val > 0.05:
                X.drop(columns = [feature], inplace = True)
                REMOVED_FEATURES[feature] = f"high p_val for chi2 contingency to {feature_name}, likely dependency {p_val}"
    return

def feature_selection(X):

    for feature in X.columns:
        #test both for variance
        if feature in REMOVED_FEATURES:
            continue

        if test_variance(X, feature):
            continue

        if METADATA[feature].type == "continuous":
            #test correlation
            test_correlation(X, feature)
        elif METADATA[feature].type == "categorical":
            test_chi_square(X, feature)    
        
        """
        if METADATA[feature].type == "continuous":
            #continuous

        elif METADATA[feature].type == "categorical":
            #categorical
        else:
            #error
        """

def evaulation(models, X, y, type):
    from sklearn.model_selection import cross_validate

    best_model = None
    best_name = None
    best_score = -np.inf

    for name, model, in tqdm(models.items(), desc = "Evaluating Models"):
        if type == "categorical":
            results = cross_validate(
                model,
                X,
                y,
                cv = 5,
                scoring = ["accuracy", "f1"]
            )

            accuracy = results["test_accuracy"].mean()
            f1 = results["test_f1"].mean()

            print(f"\n{name}")
            print(f"Accuracy:{accuracy:.4f}")
            print(f"F1 Score: {f1:.4f}\n")

            score = f1
        else:

            results = cross_validate(
                model,
                X,
                y,
                cv = 5,
                scoring = ["r2", "neg_mean_squared_error"]
            )

            r2 = results["test_r2"].mean()
            mse = -results["test_neg_mean_squared_error"].mean()

            print(f"\n{name}")
            print(f"R^2: {r2:.4f}")
            print(f"MSE: {mse:.4f}\n")

            score = r2
        
        if score > best_score:
            best_score = score
            best_model = model
            best_name = name
    
    return best_model, best_name, best_score

def modeling(X, y, type):
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import cross_val_score

    models = {}
    type_of_model = None

    if type == "continuous":
        #we train regression models for a continuous target
        #ensure numeric values
        y = pd.to_numeric(y, errors = 'coerce')
        models = {
            "Linear Regression": LinearRegression(),
            "Regressor Decision Tree": DecisionTreeRegressor(),
            #limiting so it's not too in-depth for minimal performance improvement
            "Regressor Random Forest": RandomForestRegressor(max_depth = 20, n_estimators= 25)
        }
        type_of_model = "regression"
    else:
        y = target_categorical_preprocessing(y)
        models = {
            "Logistic Regression": LogisticRegression(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier()
        }
        type_of_model = "classification"
    
    best_model = None
    score = None
    model_name = None
    try:
        best_model, model_name, score = evaulation(models, X, y, type)
    except Exception as e:
        print(f"Error evaluating models: {e}")

    return best_model, type_of_model, model_name, score

# I need to define a usability function properly- it needs to check to see if there are any values that are unusuable in a categorical or continuous feature
def usability(data, feature_name):

    if feature_name in MANUAL_OVERRIDE_TOKENS:
        return True

    #if there is no real metadata, we can do nothing with the information realistically
    if feature_name in GARBAGE_TOKENS:
        REMOVED_FEATURES[feature_name] = "feature has data not correlated to label"
        return False;
    
    #high missingness...
    missing_values = data[feature_name].isna().sum()
    if missing_values / len(data) > .5:
        REMOVED_FEATURES[feature_name] = "feature has too many missing values"
        return False

    #we don't want to use datetime or id for a learning model for now. whole different type of data that needs to be parsed in a new way
    for rfn in PREDEFINED_REMOVE_FLAGS:
        if rfn in feature_name and rfn not in MANUAL_OVERRIDE_TOKENS:
            REMOVED_FEATURES[feature_name] = "feautre is predetermined to be unreliable"
            return False

    total_entries = data[feature_name].notnull().sum()
    #no mixed types within the same column, too difficult to parse at this moment.
    numeric_count = pd.to_numeric(data[feature_name], errors = 'coerce').notnull().sum()
    ratio = numeric_count/total_entries
    if ratio < 1.0 and ratio > 0.0:
        REMOVED_FEATURES[feature_name] = "feature has mixed types ~ likely inconclusive or muddled result from machine evaluation alone"
        return False
    
    return True

#determine type is going to be a lot harder than I initially think
def initial_type_prediction(series):
    #the idea is that if this number_check comes out as anything other than 1.0 as a ratio, then there are strings
    temp = pd.to_numeric(series, errors = 'coerce')
    number_ratio = temp.notna().mean()
    #this is pretty much the only guaranteed way I have a categorical output
    if number_ratio != 1.0:
        #it's strings so there's no chance of numerical outputs... in theory
        return 'categorical'
    #not even a guarantee in this scenario.
    #finding decimals
    if (temp%1 != 0).any():
        return 'continuous'
    #no string, only integers, which means it could be categorical or not...
    total_count = series.notnull().sum()
    #cardinality check- in theory if we have a lot of categories then it should be continuous most likely
    if temp.nunique()/total_count > 0.9:
        return 'continuous'
    return 'categorical'

def encode_categorical(data, feature):
    #in this function we need to be able to modify the categorical values to be readable to a machine learning algorithm.
    #in order to do this, we just need to know the cardinality first
    cardinality = data[feature].nunique()
    if cardinality == 1:
        #only one unique value, meaning there is likely no real important information to gain
        data.drop(columns = feature, inplace = True)
        REMOVED_FEATURES[feature] = 'cardinality of 1, no unique values'
    elif cardinality == 2:
        #binary encoding
        data[feature], _ = data[feature].factorize()
    elif cardinality > 2 and cardinality < 100:
        #one-hot encoding
        old_columns = set(data.columns)
        one_hot = pd.get_dummies(data, columns = [feature], dtype = int)
        new_columns = set(one_hot) - old_columns
        METADATA.pop(feature)
        for col in new_columns:
            metadata = feature_metadata(col)
            metadata.type = "categorical"
            METADATA[col] = metadata
        return one_hot
    else:
        #for very large values, we will use frequency encoding 
        freq_map = data[feature].value_counts(normalize = True).to_dict()
        data[feature] = data[feature].map(freq_map)
    return data

def continuous_data_normalization(data, feature):
    data[feature] = pd.to_numeric(data[feature], errors = 'coerce')
    skew = data[feature].skew()
    if abs(skew) > 0.5:
        data[feature] = data[feature].fillna(data[feature].median())
    else:
        data[feature] = data[feature].fillna(data[feature].mean())    
    normalization(data, feature)

#initially cleaning the data and categorizing the features based on their immediate declarations through defined metadata
def initial_clean(data):
    for feature in tqdm(data.columns, desc = "Cleaning Data"):
        #get rid of garbage tokens, replace with nan values if they exist
        data[feature] = data[feature].astype(str).str.strip()
        data[feature] = data[feature].replace(GARBAGE_TOKENS, np.nan)
        
        #test to see if the feature is usable
        if not usability(data, feature):
            data.drop(columns = feature, inplace = True)
            continue
        
        metadata = feature_metadata(feature)
        METADATA[feature] = metadata

        #determine whether it is categorical or continuous
        if initial_type_prediction(data[feature]) == 'continuous':
            #continuous
            METADATA[feature].type = "continuous"
            continuous_data_normalization(data, feature)
        else:
            #categorical
            METADATA[feature].type = "categorical"
            data = encode_categorical(data, feature)
        #METADATA[feature].show_metadata()
    return data

#predefined print for removed data- properly formatted
def print_removed():
    print("==========================================")
    for feature in REMOVED_FEATURES:
        print(f"[XXX]REMOVED: {feature}\nREASON: {REMOVED_FEATURES[feature]}\n")
    print("==========================================")
    return

def print_metadata():
    print("==========================================")
    for feature in METADATA:
        METADATA[feature].show_metadata()
    print("==========================================")


def main():    
    csv_data = load_data()
    target_column = "performance_score"
    X = csv_data.copy()

    #get rid of all nan values
    X[target_column] = X[target_column].astype(str).str.strip()
    X[target_column] = X[target_column].replace(GARBAGE_TOKENS, np.nan)

    X.dropna(subset = [target_column])

    y = X.pop(target_column)

    X = initial_clean(X)

    p = X.shape[1]
    n = X.shape[0]
    ratio = p/n

    print(ratio)

    high_dimensionality = (p > 300) or (ratio > .01)

    if high_dimensionality:
        print("[][][]WARNING: Relatively high computational cost detected, processing time expected to be increased")
    target_type = initial_type_prediction(y)

    best_model = None
    #initial run with initial cross-validation score
    #best_model, model_type, model_name, score = modeling(X, y, target_type)

    #print(f"Type: {model_type} | Mode; Name: {model_name} | score: {score}")
    
    #feature_selection(X)
    #new_model, new_model_type, new_model_name, new_score = modeling(X, y, target_type)

    #print(f"Type: {new_model_type} | Mode; Name: {new_model_name} | score: {new_score}")
    
main()

    