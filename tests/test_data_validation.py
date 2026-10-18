
import pandas as pd
import pytest

from src.components.data_validation import DataValidation


@pytest.fixture
def valid_sales_data():
    dates = pd.date_range("2016-01-01", periods=35, freq="D")

    return pd.DataFrame({
        "id": ["ITEM_1_STORE_1"] * 35,
        "item_id": ["ITEM_1"] * 35,
        "dept_id": ["DEPT_1"] * 35,
        "cat_id": ["CAT_1"] * 35,
        "store_id": ["STORE_1"] * 35,
        "state_id": ["CA"] * 35,
        "d": [f"d_{i}" for i in range(1, 36)],
        "date": dates,
        "wm_yr_wk": [201601] * 35,
        "sales": [1] * 35,
        "sell_price": [5.0] * 35,
        "snap_CA": [0] * 35,
    })


@pytest.fixture
def validator(tmp_path):
    def make_validator(data_path):
        instance = DataValidation(data_path=data_path)
        instance.report_path = tmp_path / "validation_report.json"
        return instance

    return make_validator


def test_valid_dataset_passes(tmp_path, valid_sales_data, validator):
    data_path = tmp_path / "sales.parquet"
    valid_sales_data.to_parquet(data_path, index=False)

    result = validator(data_path).validate()

    assert result["status"] == "passed"
    assert result["rows"] == 35
    assert result["unique_series"] == 1
    assert result["unique_dates"] == 35


def test_missing_required_column_fails(tmp_path, valid_sales_data, validator):
    data_path = tmp_path / "sales.parquet"
    valid_sales_data.drop(columns=["item_id"]).to_parquet(
        data_path, index=False
    )

    with pytest.raises(ValueError, match="Missing required columns"):
        validator(data_path).validate()


def test_duplicate_series_day_fails(tmp_path, valid_sales_data, validator):
    duplicate = valid_sales_data.iloc[[0]]
    data = pd.concat([valid_sales_data, duplicate], ignore_index=True)

    data_path = tmp_path / "sales.parquet"
    data.to_parquet(data_path, index=False)

    with pytest.raises(ValueError, match="duplicate item-date"):
        validator(data_path).validate()


def test_negative_sales_fails(tmp_path, valid_sales_data, validator):
    valid_sales_data.loc[0, "sales"] = -1

    data_path = tmp_path / "sales.parquet"
    valid_sales_data.to_parquet(data_path, index=False)

    with pytest.raises(ValueError, match="negative values"):
        validator(data_path).validate()


def test_missing_input_file_fails(tmp_path, validator):
    data_path = tmp_path / "does_not_exist.parquet"

    with pytest.raises(FileNotFoundError):
        validator(data_path).validate()


def test_empty_dataset_fails(tmp_path, validator):
    data_path = tmp_path / "empty.parquet"
    pd.DataFrame().to_parquet(data_path, index=False)

    with pytest.raises(ValueError, match="empty"):
        validator(data_path).validate()
