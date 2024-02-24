from pathlib import Path
from typing import Literal

from catboost import CatBoostRegressor
from sklearn.model_selection import KFold, TimeSeriesSplit

from auxiliary import *  # noqa: F403
from logger import setup_logger
from src.train import automl_model, catboost_training_process
from src.utils import add_date_vars, reduce_to32bits

# Get repository root path
ROOT_PATH: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def intro_log(log):
    log.info("*" * 50)
    log.info("Starting script where we train for Vortex Competition With All Data.")
    log.info("*" * 50)


def prepare_environment():
    current_path = Path(ROOT_PATH)

    input_path = current_path / "input"
    # check if input path exists, if not, we will raise an error with information
    if not input_path.exists():
        raise FileNotFoundError(f"Input path {input_path} does not exist.")
    else:  # We need to check if the input path has the necessary files, train.csv and test.csv
        if not (input_path / "train.csv").exists():
            raise FileNotFoundError(f"File train.csv does not exist in {input_path}.")
        if not (input_path / "test.csv").exists():
            raise FileNotFoundError(f"File test.csv does not exist in {input_path}.")

    output_path = current_path / "output"
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def configure_logging(output_path):
    return setup_logger(file_name=f"{output_path}/all_workflow.log")


def load_data(log):
    _train_x = reduce_to32bits(get_train_x())  # noqa: F405
    _train_y = reduce_to32bits(get_train_y())  # noqa: F405
    _test_x = reduce_to32bits(get_test_x())  # noqa: F405

    log.info("Train X shape: {}".format(_train_x.shape))
    log.info("Train Y shape: {}".format(_train_y.shape))
    log.info("Test X shape: {}".format(_test_x.shape))

    return _train_x, _train_y, _test_x


def augment_data(_train_x, _test_x, log):
    X_train, added_vars = add_date_vars(_train_x)
    X_test, added_vars = add_date_vars(_test_x)
    log.info(f"Added date variables => {added_vars}.")
    return X_train, X_test


def setup_model(model_selected="catboost"):
    if model_selected == "catboost":
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
        model_selection: Literal["choose_best", "catboost"] = "catboost",
):
    assert target in [
        "U_70.0",
        "V_70.0",
        "M",
    ], "Target must be either 'U_70.0', 'V_70.0' or 'M'"
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

    log.info(f"Validation strategy: {validation_strategy} with {n_splits} splits.")

    model = setup_model()

    log.info(f">> Model selection mode => {model_selection}.")

    if model_selection == "catboost":
        X_val_target, y_val_target = catboost_training_process(
            target,
            X_train,
            _train_y,
            log,
            tscv,
            model,
            output_path,
            n_splits,
            validation_strategy,
        )

    elif model_selection == "choose_best":
        X_val_target, y_val_target = automl_model(
            target,
            X_train,
            _train_y,
            log,
            tscv,
            model,
            output_path,
            n_splits,
            validation_strategy,
        )

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
