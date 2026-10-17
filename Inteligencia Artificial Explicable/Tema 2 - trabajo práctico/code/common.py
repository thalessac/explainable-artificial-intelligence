"""Shared data preparation and prediction-export helpers for the project notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


CODE_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = CODE_DIR.parent / "data"

DATA_DICTIONARY: dict[str, str] = {
    "row_id": "identifier",
    "credit_limit": "predictor",
    "sex": "predictor / audit attribute",
    "education": "predictor",
    "marital_status": "predictor",
    "age": "predictor",
    "latest_repayment_status": "predictor",
    "maximum_repayment_delay_6m": "predictor",
    "delayed_months_6m": "predictor",
    "average_bill_6m": "predictor",
    "bill_change_sep_to_apr": "predictor",
    "total_payment_6m": "predictor",
    "payment_to_bill_ratio_6m": "predictor",
    "zero_payment_months_6m": "predictor",
    "utilization_sep": "predictor",
    "repayment_assistance_plan": "predictor",
    "payment_difficulty_next_month": "target",
    "predicted_class": "submission",
}

CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "sex",
    "education",
    "marital_status",
    "repayment_assistance_plan",
)


@dataclass(frozen=True)
class DataSplit:
    """Hold raw predictor data and its stratified training/validation split."""

    predictor_columns: list[str]
    target_column: str
    X: pd.DataFrame
    y: pd.Series
    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    X_test: pd.DataFrame


@dataclass(frozen=True)
class EncodedFeatures:
    """Hold the fitted encoder and aligned encoded feature matrices."""

    encoder: OneHotEncoder
    numerical_columns: list[str]
    encoded_categorical_columns: tuple[str, ...]
    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame


def load_data(data_dir: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the labeled training data and unlabeled test data from a directory.

    Args:
        data_dir: Directory containing ``train.csv`` and ``test.csv``. Defaults
            to this project's data directory.

    Returns:
        A tuple containing the training and test dataframes, in that order.
    """
    source_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    train_data = pd.read_csv(source_dir / "train.csv")
    test_data = pd.read_csv(source_dir / "test.csv")
    return train_data, test_data


def split_data(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    data_dictionary: Mapping[str, str] = DATA_DICTIONARY,
    test_size: float = 0.3,
    random_state: int = 411,
) -> DataSplit:
    """Select predictors and make a reproducible stratified validation split.

    Args:
        train_data: Labeled examples containing predictors and the target.
        test_data: Unlabeled examples containing the predictor columns.
        data_dictionary: Mapping from column names to their project roles.
        test_size: Fraction of labeled rows assigned to validation.
        random_state: Seed used to make the split reproducible.

    Returns:
        The raw predictor data, split partitions, and selected column names.

    Raises:
        ValueError: If the data dictionary does not define exactly one target.
    """
    predictor_columns = [
        column for column, role in data_dictionary.items() if "predictor" in role
    ]
    target_columns = [
        column for column, role in data_dictionary.items() if role == "target"
    ]
    if len(target_columns) != 1:
        raise ValueError("data_dictionary must define exactly one target column")

    target_column = target_columns[0]
    X = train_data[predictor_columns]
    y = train_data[target_column]
    X_train, X_validation, y_train, y_validation = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    X_test = test_data[predictor_columns]
    return DataSplit(
        predictor_columns=predictor_columns,
        target_column=target_column,
        X=X,
        y=y,
        X_train=X_train,
        X_validation=X_validation,
        y_train=y_train,
        y_validation=y_validation,
        X_test=X_test,
    )


def encode_features(
    data_split: DataSplit,
    categorical_columns: Sequence[str] = CATEGORICAL_COLUMNS,
) -> EncodedFeatures:
    """Fit one-hot encoding on training data and combine numeric features.

    The same fitted encoder is applied to validation and test data, preventing
    category information from leaking across the training boundary.

    Args:
        data_split: Raw train, validation, and test partitions.
        categorical_columns: Predictor columns treated as categorical.

    Returns:
        The fitted encoder and three dataframes with aligned feature columns.
    """
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    category_names = list(categorical_columns)
    numerical_columns = [
        column
        for column in data_split.predictor_columns
        if column not in category_names
    ]
    train_categories = encoder.fit_transform(data_split.X_train[category_names])
    feature_names = tuple(
        str(name) for name in encoder.get_feature_names_out(category_names)
    )
    partitions = (
        (data_split.X_train, train_categories),
        (
            data_split.X_validation,
            encoder.transform(data_split.X_validation[category_names]),
        ),
        (data_split.X_test, encoder.transform(data_split.X_test[category_names])),
    )
    encoded_partitions: list[pd.DataFrame] = []
    for raw_features, categorical_features in partitions:
        categorical_frame = pd.DataFrame(
            categorical_features,
            columns=feature_names,
            index=raw_features.index,
        )
        encoded_partitions.append(
            pd.concat([raw_features[numerical_columns], categorical_frame], axis=1)
        )

    X_train, X_validation, X_test = encoded_partitions
    return EncodedFeatures(
        encoder=encoder,
        numerical_columns=numerical_columns,
        encoded_categorical_columns=feature_names,
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test,
    )


def create_submission(
    test_data: pd.DataFrame,
    predictions: Sequence[object],
) -> pd.DataFrame:
    """Copy test rows and append their predicted class labels.

    Args:
        test_data: Original unlabeled test rows.
        predictions: One predicted class for each test row.

    Returns:
        A new dataframe containing the test rows and ``predicted_class``.
    """
    submission = test_data.copy()
    submission["predicted_class"] = predictions
    return submission


def save_predictions(
    submission: pd.DataFrame,
    model_name: str,
    output_dir: str | Path | None = None,
) -> Path:
    """Save a submission under a model-specific filename.

    Args:
        submission: Test rows with a ``predicted_class`` column.
        model_name: Filename-safe model identifier, such as ``decision_tree``.
        output_dir: Destination directory. Defaults to this project's data dir.

    Returns:
        The path to the written CSV file.
    """
    destination_dir = Path(output_dir) if output_dir is not None else DATA_DIR
    destination_dir.mkdir(parents=True, exist_ok=True)
    output_path = destination_dir / f"{model_name}_test_predictions.csv"
    submission.to_csv(output_path, index=False)
    return output_path
