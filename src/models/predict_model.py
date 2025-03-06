import joblib  
import pandas as pd  

def load_models(model_dir):
    """Load pre-trained machine learning models from the specified directory.

    Parameters
    ----------
    model_dir: str
        Directory where the trained models are saved.

    Returns
    -------
    dict
        A dictionary containing loaded models.
    """
    models = {}
    # Load models from the directory
    try:
        models['polynomial_2'] = joblib.load(f'{model_dir}/log_poly_2.joblib')
        models['adaboost_default'] = joblib.load(f'{model_dir}/adaboost_default.joblib')
        models['adaboost_tune6'] = joblib.load(f'{model_dir}/best_adaboost_Tune6.joblib')
        models['adaboost_grid'] = joblib.load(f'{model_dir}/best_adaboost_grid.joblib')
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Model files not found in {model_dir}. Please train models first.") from e

    return models

def make_predictions(models, new_data):
    """Make predictions on new data using pre-trained models.

    Parameters
    ----------
    models: dict
        A dictionary of loaded machine learning models.
    new_data: pd.DataFrame
        New data in a DataFrame format.

    Returns
    -------
    pd.DataFrame
        A DataFrame with predictions from each model.
    """
    predictions = {}

    # Make predictions using each model
    for model_name, model in models.items():
        predictions[model_name] = model.predict_proba(new_data)[:, 1]

    # Create a DataFrame with predictions
    predictions_df = pd.DataFrame(predictions)

    return predictions_df

if __name__ == "__main__":
    # Specify the directory where the trained models are saved
    model_directory = 'models/'

    # Load pre-trained models
    loaded_models = load_models(model_directory)

    # Load new data for prediction (replace with your data loading code)
    new_data = pd.read_csv('test.csv')  # Example: Loading from a CSV file

    # Make predictions on the new data
    predictions = make_predictions(loaded_models, new_data)

    # Print or save the predictions as needed
    print(predictions)
