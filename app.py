import pandas as pd
import os
import numpy as np
from scipy.stats import pearsonr

def categorical_preprocessing(dataframe, feature, unique_values):
    #make them numerical for our modeling
    swap_map = {
        value: i
        for i, value in enumerate(unique_values)
    }
    #binary encoding, nominal or ordinance does not really matter
    if len(unique_values) <= 2:
        try:
            dataframe[feature] = dataframe[feature].map(swap_map)
            return 1
        except Exception as e:
            print(f"Error mapping {feature}: {e}")
            return 0
    #one-hot encoding for more than one category
    else:
        for value in unique_values:
            dataframe[f"{feature}_{value}"] = (dataframe[feature] == value).astype(int)
        dataframe.drop(columns = [feature], inplace = True)


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
            dataframe = pd.concat([dataframe, cur_df], ignore_index = True)
    return dataframe

def usability_check(dataframe, feature):
    numeric_count = pd.to_numeric(dataframe[feature], errors = 'coerce').notnull().sum()
    total_count = dataframe[feature].notnull().sum()
    #if the ratio of numeric values is either 100% or 0%, then we can say that it is either continuous or categorical, respectively. Otherwise, we have a mixed type and we need to do some more work to determine which one it is.
    print(numeric_count)
    print(total_count)
    return numeric_count / total_count >= 1.0 or numeric_count / total_count <= 0.0

def test_continuous(dataframe, feature):
    #first let's test if the values themselves are numeric
    try:
        dataframe[feature] = pd.to_numeric(dataframe[feature], errors = 'coerce')
    except Exception as e:
        return False
    #ratio test
    n_unique = dataframe[feature].nunique()
    ratio = n_unique / len(dataframe)
    if n_unique < 10 or ratio < 0.05:
        return False
    #diff test
    if pd.api.types.is_integer_dtype(dataframe[feature]):
        diffs = np.diff(sorted(dataframe[feature].unique()))
        if len(np.unique(diffs)) <= 3:
            return False
    #entropy test (continuous features should have higher entropy)    
    from scipy.stats import entropy
    counts = dataframe[feature].value_counts()
    prob = counts / counts.sum()
    if entropy(prob) < 1.0:
        return False
    return True

def preprocess(dataframe):
    #preprocessing
    target_column = "classification"
    target_frame = dataframe[target_column]
    #features that need specification
    tbd_list = []
    if target_column not in dataframe.columns:
        print("no output")
    #it's more likely that we need some kind of human intervention here.
    #dataframe.drop([target_column], axis = 1, inplace = True)

    #we want to get rid of as many garbage tokens as possible, and then we will determine whether or not the feature is categorical or continuous
    GARBAGE_TOKENS = ['?', 'NA', 'N/A', 'null', 'None', 'nan', 'NaN', '']

    for feature in dataframe.columns:
        print(f"{feature}: {dataframe[feature].dtype}")

        dataframe[feature] = dataframe[feature].astype(str).str.strip()
        dataframe[feature] = dataframe[feature].replace(GARBAGE_TOKENS, np.nan)
        
        #now we can determine whether or not we have mixed types of numerical and categorical data
        if not usability_check(dataframe, feature):
            #mixed type would be very difficult to preprocess, so we will just drop it for now. 
            dataframe[feature].drop(columns = feature)
            continue
        #now we need to determine whether or not it is continuous or categorical
        #we are only testing whether or not it is categorical or continuous
        continuous = test_continuous(dataframe, feature)  

        if continuous:
            try:
                dataframe[feature] = pd.to_numeric(dataframe[feature], errors = 'coerce')
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
    return dataframe

"""
def feature_selection(preprocessed_dataframe):
    #now I need to do a correlation test to determine whether or not 
    legend = {}
    for feature in preprocessed_dataframe.drop(columns = ['classification'], axis = 1).columns:
        corr_val = pearsonr(preprocessed_dataframe[feature], preprocessed_dataframe['classification'])[0]
        legend[feature] = corr_val
    
    print(legend)
    return
"""

def modeling():
    return

def main():
    return


dataframe = load_data()
df = preprocess(dataframe)
print(df)