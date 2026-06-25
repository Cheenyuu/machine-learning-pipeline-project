import pandas as pd
import os
import numpy as np
from scipy.stats import pearsonr

# GLOBAL VALUES
GARBAGE_TOKENS = ['?', 'NA', 'N/A', 'null', 'None', 'nan', 'NaN', '']

def categorical_preprocessing(dataframe, feature, unique_values):
    #binary encoding, nominal or ordinance does not really matter
    if len(unique_values) <= 2:
        try:
            dataframe[feature], _ = dataframe[feature].factorize()
            return 1
        except Exception as e:
            print(f"Error mapping {feature}: {e}")
            return 0
    #one-hot encoding for more than two categories
    else:
        for value in unique_values:
            try:
                dataframe[f"{feature}_{value}"] = (dataframe[feature] == value).astype(int)
            except Exception as e:
                print(f"Error creating one-hot column for {feature}_{value}: {e}")
                return 0
        dataframe.drop(columns = [feature], inplace = True)
        return 1

def target_categorical_preprocessing(target_column):
    try:
        target_column, _ = target_column.factorize()
        return target_column
    except Exception as e:
        print(f"Error converting target column to integer value")

def normalization(dataframe, feature):
    #with the dataframe and the feature, I need to be able to normalize
    series = dataframe[feature]
    min_val = series.min()
    max_val = series.max()
    if min_val == max_val:
        try:
            dataframe[feature] = pd.Series([0.0] * len(series), index = series.index)
        except Exception as e:
            print(f"Error normalizing {feature}: {e}")
            return 1
    try:
        dataframe[feature] = (series - min_val) / (max_val - min_val)
        return 1
    except Exception as e:
        print(f"Error normalizing {feature}: {e}")
        return 0

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

def mixed_types(dataframe, feature):
    numeric_count = pd.to_numeric(dataframe[feature], errors = 'coerce').notnull().sum()
    total_count = dataframe[feature].notnull().sum()
    #if the ratio of numeric values is either 100% or 0%, then we can say that it is either continuous or categorical, respectively. Otherwise, we have a mixed type and we need to do some more work to determine which one it is.
    return numeric_count / total_count >= 1.0 or numeric_count / total_count <= 0.0

def test_continuous(series):
    #first let's test if the values themselves are numeric
    try:
        temp = pd.to_numeric(series, errors = 'coerce')
    except Exception as e:
        return False
    #ratio test
    n_unique = temp.nunique()
    ratio = n_unique / len(series)
    if n_unique < 10 or ratio < 0.05:
        return False
    #diff test
    if pd.api.types.is_integer_dtype(temp):
        diffs = np.diff(sorted(temp.unique()))
        if len(np.unique(diffs)) <= 3:
            return False
    #entropy test (continuous features should have higher entropy)    
    from scipy.stats import entropy
    counts = temp.value_counts()
    prob = counts / counts.sum()
    if entropy(prob) < 1.0:
        return False
    return True


# I need to define a usability function properly- it needs to check to see if there are any values that are unusuable in a categorical or continuous feature
def usability(feature_name, type):

    #if there is no real metadata, we can do nothing with the information realistically
    if feature_name in GARBAGE_TOKENS:
        return False;
    
    if type == 'continuous':
        
        
        return
    else:
        #categorical data
        return


def preprocess(dataframe):
    #preprocessing
    
    #get rid of any constant columns
    constant_cols = [col for col in dataframe.columns if dataframe[col].nunique() <= 1]
    dataframe.drop(columns = constant_cols, inplace = True)
    
    #we want to get rid of as many garbage tokens as possible, and then we will determine whether or not the feature is categorical or continuous
    GARBAGE_TOKENS = ['?', 'NA', 'N/A', 'null', 'None', 'nan', 'NaN', '']

    for feature in dataframe.columns:
        
        dataframe[feature] = dataframe[feature].astype(str).str.strip()
        dataframe[feature] = dataframe[feature].replace(GARBAGE_TOKENS, np.nan)
        
        #now we can determine whether or not we have mixed types of numerical and categorical data
        if not mixed_types(dataframe, feature):
            #mixed type would be very difficult to preprocess, so we will just drop it for now. 
            dataframe[feature].drop(columns = feature)
            continue

        #now we need to determine whether or not it is continuous or categorical
        #we are only testing whether or not it is categorical or continuous
        continuous = test_continuous(dataframe[feature])  

        if continuous:
            try:
                dataframe[feature] = pd.to_numeric(dataframe[feature], errors = 'coerce')
                
                #I need to move this to the usability function, we do not need any features with excess in nan values...
                dataframe[feature] = dataframe[feature].fillna(dataframe[feature].mean())
                
                normalization(dataframe, feature)
            except Exception as e:
                print(f"Error processing {feature} as continuous: {e}")
        else:
            try:
                #we need to test if it's all unique values, in which case we can just drop it. Otherwise, we can do categorical preprocessing.
                if dataframe[feature].nunique() == dataframe[feature].notnull().sum():
                    dataframe.drop(columns = feature, inplace = True)
                    continue
                dataframe[feature] = dataframe[feature].fillna(dataframe[feature].mode()[0])
                categorical_preprocessing(dataframe, feature, dataframe[feature].dropna().unique())
            except Exception as e:
                print(f"Error processing {feature} as categorical: {e}")
        #by the end, we know everything should be numerical, so we can test the variance of the entire dataset to determine whether or not we need to 
        #drop those columns
        threshold = 0.01
        #variance = dataframe[feature].var()

    threshold = 0.01
    low_var_cols = [col for col in dataframe.columns if dataframe[col].var() < threshold]
    try:
        dataframe.drop(columns = low_var_cols, inplace = True)
    except Exception as e:
        print(f"Error dropping low variance columns: {e}")
    return dataframe


def feature_selection(preprocessed_dataframe, y, continuous):
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    preprocessed_dataframe["random_noise"] = np.random.randn(len(preprocessed_dataframe))
    #[True, False, ..., True] for every feature, define whether or not it is discrete or not
    is_discrete = preprocessed_dataframe.dtypes == int
    if(continuous):
        mi = mutual_info_regression(preprocessed_dataframe, y, discrete_features = is_discrete)
    else:
        mi = mutual_info_classif(preprocessed_dataframe, y, discrete_features = is_discrete)
    mi = pd.Series(mi, index = preprocessed_dataframe.columns, name = "mutual_info")
    mi = mi.sort_values(ascending = False)
    threshold = mi["random_noise"]
    drop_features = mi[mi <= threshold].index.tolist()
    preprocessed_dataframe.drop(columns = drop_features, inplace = True)
    
    return


def evaulation(models, X, y, continuous):
    from sklearn.model_selection import cross_val_score
    scoring = 'r2'
    if continuous:
        scoring = 'f1'
    model_name = None
    best_model = None
    best_score = float("-inf")
    print("Models Evaluated:")
    print("------------------------------")
    for model in models:
        score = cross_val_score(models[model], X, y, cv = 5, scoring = scoring).mean()
        print(f"{model}")
        print(f"Cross validation score mean: {score}\n\n")
        if score > best_score:
            best_score = score
            best_model = models[model]
            model_name = model
    print(f"Selected model: {model_name}")
    return best_model

def modeling(X, y, continuous):
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import cross_val_score

    models = {}
    type_of_model = None

    if continuous:
        #we train regression models for a continuous target
        #ensure numeric values
        y = pd.to_numeric(y, errors = 'coerce')
        models = {
            "Linear Regression": LinearRegression(),
            "Regressor Decision Tree": DecisionTreeRegressor(),
            "Regressor Random Forest": RandomForestRegressor()
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
    try:
        best_model = evaulation(models, X, y, continuous)
    except Exception as e:
        print(f"Error evaluating models: {e}")

    return best_model, type_of_model



def main():    
    csv_data = load_data()
    target_column = "performance_score"
    X = csv_data.copy()

    """
    #I need to preprocess some of the data separately for the target column
    GARBAGE_TOKENS = ['?', 'NA', 'N/A', 'null', 'None', 'nan', 'NaN', '']
    X[target_column] = X[target_column].astype(str).str.strip()
    X[target_column] = X[target_column].replace(GARBAGE_TOKENS, np.nan)

    #get rid of rows that do not have target output
    X = X.dropna(subset = [target_column])

    y = X.pop(target_column)
    preprocessed_dataframe = preprocess(X)
    number_of_features = preprocessed_dataframe.shape[1]
    continuous = test_continuous(y)
    feature_selection(preprocessed_dataframe, y, continuous)
    print(F"features: {preprocessed_dataframe.columns.tolist()}")
    print(f"Removed {number_of_features - preprocessed_dataframe.shape[1]} features, kept {preprocessed_dataframe.shape[1]} features.")
    model = modeling(preprocessed_dataframe, y, continuous)
    """

main()