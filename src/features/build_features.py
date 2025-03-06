import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def drop_columns(data, columns_to_drop):
    """Drop specified columns from the DataFrame."""
    return data.drop(columns=columns_to_drop, axis=1)

def replace_missing_values(data, columns_to_replace):
    """Replace missing values in specified columns with 0."""
    data[columns_to_replace] = data[columns_to_replace].fillna(0)
    return data

def handle_missing_values(data):
    """Handle missing values by dropping rows with missing values."""
    return data.dropna()

def preprocess_numeric(data, numeric_columns):
    """Replace missing values with the mean for numeric columns."""
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())
    return data

def preprocess_categorical(data, categorical_columns):
    """Replace missing values with the mode for categorical columns."""
    for col in categorical_columns:
        mode_val = data[col].mode().iloc[0]
        data[col].fillna(mode_val, inplace=True)
    return data

def encode_categorical(data, cat_cols):
    """Label encode categorical columns."""
    le = LabelEncoder()
    for col in cat_cols:
        data[col] = le.fit_transform(data[col])
    return data

def scale_numeric(data, num_cols):
    """Standardize numeric columns."""
    scaler = StandardScaler()
    data[num_cols] = scaler.fit_transform(data[num_cols])
    return data

def balance_data(X, y):
    """Apply SMOTE oversampling to balance the target variable."""
    oversample = SMOTE()
    X, y = oversample.fit_resample(X, y)
    return X, y

if __name__ == "__main__":
    # Load data
    df_cleaned = load_data('data/train.csv')
    test_cleaned = load_data('data/test.csv')

    # Specify columns to drop
    columns_to_drop = ['pick', 'Rec_Rank']
    
    # Preprocess data
    df_cleaned = drop_columns(df_cleaned, columns_to_drop)
    test_cleaned = drop_columns(test_cleaned, columns_to_drop)
    
    # Specify columns to replace missing values with 0
    columns_to_replace = ['rimmade', 'rimmade_rimmiss', 'midmade', 'midmade_midmiss',
                          'dunksmade', 'dunksmiss_dunksmade', 'rim_ratio', 'mid_ratio', 'dunks_ratio']
    
    df_cleaned = replace_missing_values(df_cleaned, columns_to_replace)
    test_cleaned = replace_missing_values(test_cleaned, columns_to_replace)
    
    # Handle missing values
    df_cleaned = handle_missing_values(df_cleaned)
    test_cleaned = handle_missing_values(test_cleaned)
    
    # Define numeric and categorical columns
    num_cols = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = list(set(df_cleaned.columns) - set(num_cols))
    
    # Preprocess numeric and categorical data
    df_cleaned = preprocess_numeric(df_cleaned, num_cols)
    test_cleaned = preprocess_numeric(test_cleaned, num_cols)
    
    df_cleaned = preprocess_categorical(df_cleaned, cat_cols)
    test_cleaned = preprocess_categorical(test_cleaned, cat_cols)
    
    # Encode categorical columns
    df_cleaned = encode_categorical(df_cleaned, cat_cols)
    test_cleaned = encode_categorical(test_cleaned, cat_cols)
    
    # Scale numeric columns
    df_cleaned = scale_numeric(df_cleaned, num_cols)
    test_cleaned = scale_numeric(test_cleaned, num_cols)
    
    # Handle class imbalance
    target = df_cleaned.pop('drafted')
    X, y = balance_data(df_cleaned, target)

    # Split data into train, validation, and test sets
    X_data, X_test, y_data, y_test = train_test_split(X, y, test_size=0.2, random_state=8)
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.2, random_state=8)

    # Save preprocessed data
    X_train.to_csv('../data/processed/X_train.csv', index=False)
    X_val.to_csv('../data/processed/X_val.csv', index=False)
    X_test.to_csv('../data/processed/X_test.csv', index=False)
    y_train.to_csv('../data/processed/y_train.csv', index=False)
    y_val.to_csv('../data/processed/y_val.csv', index=False)
    y_test.to_csv('../data/processed/y_test.csv', index=False)
