import os
from pathlib import Path
from typing import Literal
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from automl_stuff_regression import *  # noqa: F403
from auxiliary import *  # noqa: F403
from catboost import CatBoostRegressor, Pool
from logger import setup_logger
from pycaret.regression import *  # noqa: F403
from pynational.utils.preprocessing import add_date_vars
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, train_test_split

# Assume imports for setup_logger, get_train_x, get_train_y, get_test_x, add_date_vars, setup, create_model, save_model, load_model, compare_models, pull, get_best_models_by_metric, train_blender_based_on_best_metrics are correctly in place.

def intro_log(log):
    log.info("*" * 50)
    log.info(
        "Starting script where we train CatBoost Vortex Comptetition With All Data."
    )
    log.info("*" * 50)

def prepare_environment():
    current_path = Path(os.getcwd())
    output_path = current_path / "output"
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def configure_logging(output_path):
    return setup_logger(file_name=f"{output_path}/catboost.log")


def load_data(log):
    _train_x = get_train_x()  # noqa: F405
    _train_y = get_train_y()  # noqa: F405
    _test_x = get_test_x()  # noqa: F405

    log_process(log, _train_x, _train_y, _test_x)

    return _train_x, _train_y, _test_x


def log_process(log, _train_x, _train_y, _test_x):
    log.info("Train X shape: {}".format(_train_x.shape))
    log.info("Train Y shape: {}".format(_train_y.shape))
    log.info("Test X shape: {}".format(_test_x.shape))


def augment_data(_train_x, _test_x, log):
    X_train, added_vars = add_date_vars(_train_x)
    X_test, added_vars = add_date_vars(_test_x)
    log.info("Added date variables.")
    return X_train, X_test


def setup_model():
    return CatBoostRegressor(
        iterations=100000,
        verbose=500,
        task_type="GPU",
        devices="0:1",
        random_seed=44,
        use_best_model=True,
        early_stopping_rounds=200,
        border_count=254,
        grow_policy="SymmetricTree",
        allow_writing_files=False,
    )


def train_model_for_target(
    X_train,
    _train_y,
    log,
    output_path,
    n_splits,
    target: Literal["U_70.0", "V_70.0", "M"],
    validation_strategy: Literal["TimeSeriesSplit", "KFold"] = "TimeSeriesSplit",
):
    assert target in ["U_70.0", "V_70.0", "M"], "Target must be either 'U_70.0', 'V_70.0' or 'M'"
    assert validation_strategy in [
        "TimeSeriesSplit",
        "KFold",
    ], "Validation strategy must be either 'TimeSeriesSplit' or 'KFold'"
    assert isinstance(n_splits, int), "n_splits must be an integer"

    if validation_strategy == "TimeSeriesSplit":
        tscv = TimeSeriesSplit(n_splits=n_splits)
    else:
        # Create an instance of KFold
        tscv = KFold(n_splits=n_splits, shuffle=False)
        
    log.info(f'Validation strategy: {validation_strategy} with {n_splits} splits.')

    model = setup_model()

    X_val_target, y_val_target = model_training_process(
        target, X_train, _train_y, log, tscv, model, output_path, n_splits, validation_strategy
    )
    return X_val_target, y_val_target


def model_training_process(
    TARGET, X_train, _train_y, log, tscv, model, output_path, n_splits, validation_strategy
):
    log.info(f"Training model for {TARGET}...")

    log.info("Splitting into train and validation sets...")
    
    if TARGET == "M":
        # Calculate M from U and V
        _train_y["M"] = np.sqrt(_train_y["U_70.0"] ** 2 + _train_y["V_70.0"] ** 2)
    
    X_train_target, X_val_target, y_train_target, y_val_target = train_test_split(
        X_train, _train_y[[TARGET]], test_size=0.3, shuffle=False
    )

    model_path = output_path / f"{TARGET}" / validation_strategy
    model_path.mkdir(parents=True, exist_ok=True)

    train_scores = []
    val_scores = []

    for fold, (train_index, val_index) in enumerate(tscv.split(X_train_target), 1):
        cbm_path = os.path.join(model_path, f"model_fold_{fold}.cbm")

        # Is the last fold?
        is_last_fold = fold == n_splits

        X_train_fold = X_train_target.iloc[train_index]
        X_val_fold = X_train_target.iloc[val_index]
        y_train_fold = y_train_target.iloc[train_index]
        y_val_fold = y_train_target.iloc[val_index]

        train_pool = Pool(X_train_fold, y_train_fold)
        val_pool = Pool(X_val_fold, y_val_fold)

        # Print the shape of the training and validation set
        log.info(
            f"Fold {fold}: Train shape: {X_train_fold.shape}, Val shape: {X_val_fold.shape}"
        )
        
        if os.path.exists(cbm_path):
            log.info(f"Model for fold {fold} already exists. Skipping...")
            continue
        
        model.fit(train_pool, eval_set=val_pool)

        log.info(f"Model trained for fold {fold}.")
        model.save_model(cbm_path)
        log.info(f"Model saved at {cbm_path}.")

        if is_last_fold:
            # get the best iteration
            best_iteration = model.get_best_iteration()
            # get params
            params = model.get_params()
            params["iterations"] = best_iteration
            params["use_best_model"] = False

        y_train_pred = model.predict(X_train_fold)
        y_val_pred = model.predict(X_val_fold)

        train_rmse = np.sqrt(mean_squared_error(y_train_fold, y_train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val_fold, y_val_pred))

        log.info(
            f"Target {TARGET} || Fold {fold}: Train RMSE = {train_rmse}, Val RMSE = {val_rmse}"
        )

        train_scores.append(train_rmse)
        val_scores.append(val_rmse)

    # We will train the model on the entire training data and make predictions on the test data, using the
    cbm_path = os.path.join(model_path, "final_model.cbm")
    if os.path.exists(cbm_path):
        log.info("Final model already exists. Skipping...")

    else:
        #  the average scores across all folds
        log.info(f"Average Train RMSE - {TARGET}: {np.mean(train_scores)}")
        log.info(f"Average Validation RMSE - {TARGET}: {np.mean(val_scores)}")

        if is_last_fold:
            final_model = CatBoostRegressor(**params)
            final_model.fit(X_train_target, y_train_target, verbose=500)
            final_model.save_model(cbm_path)
            log.info(f"Final model saved to {cbm_path}.")

    log.info(f"Model training for {TARGET} completed.")
    return X_val_target, y_val_target


def shut_down_pc():
    os.system("shutdown /s /t 120")


def main():
    output_path = prepare_environment()
    log = configure_logging(output_path)
    intro_log(log)

    _train_x, _train_y, _test_x = load_data(log)

    # X_train is training and validation set
    # X_test is the test set to be predicted por competition
    X_train, X_test = augment_data(_train_x, _test_x, log)

    # Train the model for each target in the training set
    # Use validation to avoid overfitting and to get the best model
    targets = ["U_70.0", "V_70.0", "M"]
    for target in targets:
        for validation_strategy in ["TimeSeriesSplit", "KFold"]:
            X_val, y_val = train_model_for_target(
                X_train,
                _train_y,
                log,
                output_path,
                n_splits=8,
                target=target,
                validation_strategy=validation_strategy,
            )

        try:
            y_val.rename(columns={target: "target"}, inplace=True)
            # Concat X_val and y_val and save it to parquet
            pd.concat([X_val, y_val], axis=1).to_parquet(
                output_path / target / f"validation_{target}.parquet"
            )
        except ValueError as e:
            log.error(f"Error: {e}")
            print()
            print(type(y_val))
            print(type(X_val))
            print()
            print(X_val.shape)
            print(y_val.shape)
            print()
            print(X_val.head())
            print(y_val.head())
            raise e

    # TODO: Train the blender model using Optuna, ir order to reduce Objective Function

    log.info("All models trained and saved.")
    log.info("Script completed.")
    log.info("DONE :) by LTR")

    # Uncomment below line to enable PC shutdown after execution
    # shut_down_pc()

if __name__ == "__main__":
    main()
