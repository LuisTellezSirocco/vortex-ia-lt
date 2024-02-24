import os

import numpy as np
import pandas as pd
from automl_stuff_regression import *  # noqa: F403
from auxiliary import *  # noqa: F403
from catboost import CatBoostRegressor, Pool
from pycaret.regression import *  # noqa: F403
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def catboost_training_process(
    TARGET,
    X_train,
    _train_y,
    log,
    tscv,
    model,
    output_path,
    n_splits,
    validation_strategy,
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


def automl_model(
    TARGET,
    X_train,
    _train_y,
    log,
    tscv,
    model,
    output_path,
    n_splits,
    validation_strategy,
):
    log.info(f"Training model for {TARGET}...")

    log.info("Splitting into train and validation sets...")

    if TARGET == "M":
        # Calculate M from U and V
        _train_y["M"] = np.sqrt(_train_y["U_70.0"] ** 2 + _train_y["V_70.0"] ** 2)

    X_train_target, X_val_target, y_train_target, y_val_target = train_test_split(
        X_train, _train_y[[TARGET]], test_size=0.3, shuffle=False
    )

    # create a dataframe by concating X_train_target and y_train_target, change target column name to 'target'
    train_data = pd.concat(
        [X_train_target, y_train_target.rename(columns={TARGET: "target"})], axis=1
    )

    if validation_strategy == "TimeSeriesSplit":
        fold_strategy = "timeseries"
    else:
        fold_strategy = "kfold"

    exp = setup(  # noqa: F405
        data=train_data,
        target="target",
        session_id=42,
        log_experiment=False,
        preprocess=True,
        n_jobs=2,
        fold_strategy=fold_strategy,
        fold_shuffle=False,
        data_split_shuffle=False,
        fold=5,
    )

    best_model = compare_models(  # noqa: F405
        turbo=False,  
        n_select=1,
        exclude=["lar", "par", "dt", "omp", "llar", "ransac", "kr", "lightgbm", "ard"],
    )

    model_path = output_path / f"{TARGET}" / validation_strategy
    model_path.mkdir(parents=True, exist_ok=True)

    total_path = os.path.join(model_path, "best_model")

    save_model(best_model, total_path)  # noqa: F405

    train_scores = []
    val_scores = []

    for fold, (train_index, val_index) in enumerate(tscv.split(X_train_target), 1):
        bm_path = os.path.join(model_path, f"best_model_{fold}")

        X_train_fold = X_train_target.iloc[train_index]
        X_val_fold = X_train_target.iloc[val_index]
        y_train_fold = y_train_target.iloc[train_index]
        y_val_fold = y_train_target.iloc[val_index]

        # Print the shape of the training and validation set
        log.info(
            f"Fold {fold}: Train shape: {X_train_fold.shape}, Val shape: {X_val_fold.shape}"
        )

        if os.path.exists(bm_path):
            log.info(f"Model for fold {fold} already exists. Skipping...")
            continue

        # create a dataframe by concating X_train_target and y_train_target, change target column name to 'target'
        train_data = pd.concat(
            [X_train_fold, y_train_fold.rename(columns={TARGET: "target"})], axis=1
        )

        exp = setup(  # noqa: F405
            data=train_data,
            target="target",
            session_id=42,
            log_experiment=False,
            preprocess=True,
            n_jobs=2,
            fold_strategy=fold_strategy,
            fold_shuffle=False,
            data_split_shuffle=False,
            fold=5,
        )

        model = create_model(best_model, cross_validation=False)  # noqa: F405

        log.info(f"Model trained for fold {fold}.")

        save_model(model, bm_path)  # noqa: F405
        log.info(f"Model saved at {bm_path}.")

        y_train_pred = model.predict(X_train_fold)
        y_val_pred = model.predict(X_val_fold)

        train_rmse = np.sqrt(mean_squared_error(y_train_fold, y_train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val_fold, y_val_pred))

        log.info(
            f"Target {TARGET} || Fold {fold}: Train RMSE = {train_rmse}, Val RMSE = {val_rmse}"
        )

        train_scores.append(train_rmse)
        val_scores.append(val_rmse)

    #  the average scores across all folds
    log.info(f"Average Train RMSE - {TARGET}: {np.mean(train_scores)}")
    log.info(f"Average Validation RMSE - {TARGET}: {np.mean(val_scores)}")

    log.info(f"Model training for {TARGET} completed.")
    return X_val_target, y_val_target
