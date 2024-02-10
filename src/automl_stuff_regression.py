import pandas as pd
from pycaret.regression import *


def create_lag_and_lead(df, variables):
    """
    Crea variables con el valor de la variable en el paso anterior y posterior.

    Parámetros
    ----------
        df : pd.DataFrame
            dataframe con los datos
        variables : list
            lista de variables para las que crear las variables con el valor anterior y posterior

    Devuelve
    --------
        pd.DataFrame : pd.dataframe
            df con las variables añadidas
    """
    df_copy = df.copy()

    new_columns = []
    for var in variables:
        if var in df_copy.columns:
            new_columns.append(df_copy[var].shift(-1).rename(var + '_lag1'))
            new_columns.append(df_copy[var].shift(1).rename(var + '_lead1'))
        else:
            raise ValueError(f'La variable {var} no está en el dataframe.')

    return pd.concat([df_copy] + new_columns, axis=1)


def get_best_models_by_metric(result_metrics):
    """
    Identify the best model for each metric from the result_metrics DataFrame.

    Parameters:
        result_metrics (pd.DataFrame): A DataFrame containing model performance metrics.

    Returns:
        dict: A dictionary with metric names as keys and corresponding best model names as values.
    """
    # Dictionary to hold the best model for each metric
    best_models = {}

    # Metrics to consider for finding the best model
    metrics = ['MAE', 'MSE', 'RMSE', 'R2', 'RMSLE', 'MAPE']

    # Loop through each metric and find the model with the best score
    for metric in metrics:
        # For MAE, MSE, RMSE, RMSLE, MAPE lower is better, for R2 higher is better
        if metric == 'R2':
            best_models[metric] = result_metrics.loc[result_metrics[metric].idxmax()]['Model']
        else:
            best_models[metric] = result_metrics.loc[result_metrics[metric].idxmin()]['Model']

    return best_models


def get_model_id(name):
    """
    Get the PyCaret internal ID for a given model name.

    Parameters:
        name (str): The name of the model.
        models_df (pd.DataFrame): DataFrame containing models information from PyCaret's models function.

    Returns:
        str: The PyCaret internal ID for the model.
    """
    models_df = models()

    name = name.lower()
    model_id = models_df[models_df['Name'].str.lower() == name].index[0]
    return model_id


def train_blender_based_on_best_metrics(best_models_dict):
    """
    Train a blender model based on the best models for each metric.

    Parameters:
        best_models_dict (dict): Dictionary with metric names as keys and corresponding best model names as values.
        models_df (pd.DataFrame): DataFrame containing models information from PyCaret's models function.

    Returns:
        model: The trained and finalized blender model.
    """
    # Create a mapping of model names to PyCaret IDs
    model_ids = {metric: get_model_id(name) for metric, name in best_models_dict.items()}

    # Count the occurrences of each model ID to assign weights
    model_counts = {model_id: list(model_ids.values()).count(model_id) for model_id in
                    model_ids.values()}

    # Initialize the list to hold the model objects and their weights
    models_with_weights = []

    # Create and add models to the list with corresponding weights
    print('Creating models with the following IDs and weights:')
    for model_id, weight in model_counts.items():
        print(f'Creating model with ID: {model_id} and weight: {weight}')
        model = create_model(model_id, verbose=False)
        models_with_weights.append((model, weight))

    # Create blender model using the list of models with weights
    blender = blend_models(estimator_list=[model[0] for model in models_with_weights],
                           weights=[model[1] for model in models_with_weights])

    # Finalize the blender model
    final_blender = finalize_model(blender)

    # Return the finalized blender model
    return final_blender

# Usage example
# import train_test_split from sklearn and apply it for splitting the data 80-20 as a time series
# from sklearn.model_selection import train_test_split
#
# train, test = train_test_split(inicio, test_size=0.2, shuffle=False)
#
# # print shape of train and test data
# print('Train Shape:', train.shape)
# print('Test Shape:', test.shape)
#
# exp = setup(
#     data=train,
#     target='obs',
#     session_id=42,
#     log_experiment=False,
#     preprocess=True,
#     n_jobs=3,
#     fold_strategy="timeseries",
#     fold_shuffle=False,
#     data_split_shuffle=False,
#     fold=5,
# )
#
# best_model = compare_models(turbo=False)
#
# result_metrics = pull()
#
# best_model_by_metric_dict = get_best_models_by_metric(result_metrics)
#
# blender = train_blender_based_on_best_metrics(best_model_by_metric_dict)
#
# preds = blender.predict(test.drop(columns=['obs']))
