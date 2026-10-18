import numpy as np
import pandas as pd
import pytest

from src.components.model_evaluation import ModelEvaluator


def test_calculate_metrics():
    y_true = np.array([10, 20, 30])
    y_pred = np.array([12, 18, 33])

    metrics = ModelEvaluator.calculate_metrics(y_true, y_pred)

    assert metrics["mae"] == pytest.approx(7 / 3)
    assert metrics["rmse"] == pytest.approx(np.sqrt(17 / 3))
    assert metrics["wape_percent"] == pytest.approx(7 / 60 * 100)
    assert metrics["mean_bias"] == pytest.approx(1.0)
    assert metrics["n_samples"] == 3


def test_perfect_predictions_have_zero_errors():
    values = np.array([0, 5, 10, 20])

    metrics = ModelEvaluator.calculate_metrics(values, values)

    assert metrics["mae"] == 0
    assert metrics["rmse"] == 0
    assert metrics["wape_percent"] == 0
    assert metrics["mean_bias"] == 0


def test_wape_is_none_when_actual_sales_sum_to_zero():
    metrics = ModelEvaluator.calculate_metrics(y_true=[0, 0, 0], y_pred=[0, 1, 2],)

    assert metrics["wape_percent"] is None
    assert metrics["mae"] == pytest.approx(1.0)


def test_mismatched_array_lengths_are_rejected():
    with pytest.raises(ValueError, match="equal"):
        ModelEvaluator.calculate_metrics(y_true=[1, 2, 3], y_pred=[1, 2])


def test_empty_arrays_are_rejected():
    with pytest.raises(ValueError, match="non-zero"):
        ModelEvaluator.calculate_metrics([], [])


@pytest.mark.parametrize(
    "actual,predicted",
    [([1, np.nan], [1, 2]),
     ([1, 2], [1, np.inf]),
    ]
)
def test_non_finite_values_are_rejected(actual, predicted):
    with pytest.raises(ValueError, match="non-finite"):
        ModelEvaluator.calculate_metrics(actual, predicted)


def test_validation_split_uses_latest_dates():
    evaluator = ModelEvaluator(validation_days=3)

    dates = pd.date_range("2016-01-01", periods=10, freq="D")

    df = pd.DataFrame({
        "date": list(dates) * 2,
        "item_id": ["ITEM_1"] * 10 + ["ITEM_2"] * 10,
        "sales": list(range(10)) * 2,
    })

    validation = evaluator.split_data(df)

    assert validation["date"].nunique() == 3
    assert validation["date"].min() == pd.Timestamp("2016-01-08")
    assert validation["date"].max() == pd.Timestamp("2016-01-10")
    assert len(validation) == 6


def test_validation_split_rejects_insufficient_dates():
    evaluator = ModelEvaluator(validation_days=3)

    df = pd.DataFrame({
        "date": pd.to_datetime(["2016-01-01", "2016-01-02", "2016-01-03"]),
        "sales": [1, 2, 3],
    })

    with pytest.raises(ValueError, match="Not enough unique dates"):
        evaluator.split_data(df)


def test_zero_validation_days_are_rejected():
    with pytest.raises(ValueError, match="must be positive"):
        ModelEvaluator(validation_days=0)
